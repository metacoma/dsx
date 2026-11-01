//! Logits-equality tests for the multi-session prefix cache on the GPU.
//!
//! They compare a host-backed park/restore, a sparse anchor rewind and an `A -> B -> A` session switch against a cold
//! prefill of the same prompt. The checkpoint is not shipped, so the tests are ignored by default; run them on a free
//! GPU after stopping the server:
//!
//! ```text
//! DSX_SESSION_CACHE_MODEL=/path/to/dsx-t16.gguf \
//!   cargo test --release --features cuda --test session_cache_logits -- --ignored --nocapture
//! ```
//!
//! `DSX_SESSION_CACHE_CTX` and `DSX_SESSION_CACHE_CHUNK` override the context and the prefill chunk.

#![cfg(feature = "cuda")]

use std::path::PathBuf;

use anyhow::{Result, ensure};
use dsx::gpumodel::{GpuModel, Placement, PrefillBufs};
use dsx::hostcache::HostCacheArena;
use dsx::model::Model;

/// Prompt tokens per lineage in the tests.
const PROMPT: usize = 256;
/// Tail tokens appended after reopening a lineage.
const TAIL: usize = 32;
/// Arena budget for the tests (parked rings + carry + used rows of a short prompt fit easily).
const ARENA_BYTES: usize = 512 << 20;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn model_path() -> PathBuf {
    std::env::var_os("DSX_SESSION_CACHE_MODEL")
        .map(PathBuf::from)
        .expect("set DSX_SESSION_CACHE_MODEL to the dsx GGUF checkpoint")
}

/// Load the checkpoint once and build the engine plus the prefill buffers that `prefill` needs.
fn engine(ctx: usize) -> Result<(&'static Model, GpuModel<'static>, PrefillBufs)> {
    let m: &'static Model = Box::leak(Box::new(Model::load(&model_path(), None)?));
    let mut gm = GpuModel::with_placement(
        m,
        -1,
        ctx,
        Placement {
            gpu0_gb: -1.5,
            aux_gb: -1.0,
            arena_gb: 0.0,
            prefill_chunk: env_usize("DSX_SESSION_CACHE_CHUNK", 512),
            rank: None,
        },
    )?;
    let pb = gm.pb.take().expect("prefill buffers");
    Ok((m, gm, pb))
}

/// Deterministic in-vocabulary token ids: no tokenizer and no special-token handling needed.
fn tokens(vocab: usize, seed: u64, n: usize) -> Vec<u32> {
    assert!(vocab > 4, "vocab too small for the token generator");
    let mut x = seed | 1;
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // keep clear of the reserved low ids
            2 + ((x >> 33) as usize % (vocab - 4)) as u32
        })
        .collect()
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i)
}

/// Compare against the cold-prefill reference. Kernels are deterministic, so the tolerance only covers a possible
/// difference in reduction order between one chunked prefill and a shorter tail prefill.
fn assert_close(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: logit count");
    let mut max = 0f32;
    let mut at = 0usize;
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        let d = (a - b).abs();
        if d > max {
            max = d;
            at = i;
        }
    }
    assert_eq!(argmax(got), argmax(want), "{label}: argmax differs");
    eprintln!("{label}: max |delta logit| = {max:e} at {at}");
    assert!(max <= 1e-3, "{label}: logits diverged, max |delta| = {max:e} at {at}");
}

/// Park a lineage, overwrite the device with another one, restore: the reopened tail must match a cold prefill.
#[test]
#[ignore = "needs the checkpoint and a free GPU"]
fn park_restore_matches_cold_prefill() -> Result<()> {
    let ctx = env_usize("DSX_SESSION_CACHE_CTX", 2048);
    let (m, mut gm, pb) = engine(ctx)?;
    let vocab = m.cfg.vocab;
    let a = tokens(vocab, 0xA11CE, PROMPT);
    let tail = tokens(vocab, 0x7A11, TAIL);

    let la = gm.prefill(&pb, &a)?;
    let mut arena = HostCacheArena::pinned(ARENA_BYTES, 32 << 20);
    let parked = gm.park_state(&mut arena, la.clone())?;
    assert_close("park stores the boundary logits", &parked.logits, &la);

    gm.reset();
    let cold = gm.prefill(&pb, &[a.clone(), tail.clone()].concat())?;

    gm.reset();
    gm.restore_state(&arena, &parked)?;
    ensure!(gm.pos == a.len(), "restored position {} != {}", gm.pos, a.len());
    let reopened = gm.prefill(&pb, &tail)?;
    assert_close("park/restore tail", &reopened, &cold);
    Ok(())
}

/// Capture a rewind point at a chunk boundary, reopen from the host, and compare the tail with a cold prefill.
#[test]
#[ignore = "needs the checkpoint and a free GPU"]
fn anchor_rewind_matches_cold_prefill() -> Result<()> {
    let ctx = env_usize("DSX_SESSION_CACHE_CTX", 2048);
    let (m, mut gm, pb) = engine(ctx)?;
    let vocab = m.cfg.vocab;
    let prompt = tokens(vocab, 0xA9C40, PROMPT * 2);
    let split = PROMPT;

    let cold = gm.prefill(&pb, &prompt)?;

    let mut arena = HostCacheArena::pinned(ARENA_BYTES, 32 << 20);
    gm.reset();
    gm.prefill(&pb, &prompt[..split])?;
    let anchor = gm.capture_rewind(&mut arena)?;
    ensure!(anchor.pos == split, "anchor pos {} != {}", anchor.pos, split);
    // keep the live sequence advancing so the rewind is not a no-op
    gm.prefill(&pb, &prompt[split..])?;

    gm.reset();
    gm.restore_rewind(&arena, &anchor)?;
    ensure!(gm.pos == split, "rewound position {} != {}", gm.pos, split);
    let reopened = gm.prefill(&pb, &prompt[split..])?;
    assert_close("anchor rewind tail", &reopened, &cold);
    Ok(())
}

/// `A -> B -> A`: park both lineages, then reopen each and compare its tail with a cold prefill.
#[test]
#[ignore = "needs the checkpoint and a free GPU"]
fn a_b_a_matches_cold_prefill() -> Result<()> {
    let ctx = env_usize("DSX_SESSION_CACHE_CTX", 2048);
    let (m, mut gm, pb) = engine(ctx)?;
    let vocab = m.cfg.vocab;
    let a = tokens(vocab, 0xA, PROMPT);
    let b = tokens(vocab, 0xB, PROMPT);
    let ta = tokens(vocab, 0xA1A, TAIL);
    let tb = tokens(vocab, 0xB1B, TAIL);

    let mut arena = HostCacheArena::pinned(ARENA_BYTES, 32 << 20);
    let la = gm.prefill(&pb, &a)?;
    let parked_a = gm.park_state(&mut arena, la)?;
    gm.reset();
    let lb = gm.prefill(&pb, &b)?;
    let parked_b = gm.park_state(&mut arena, lb)?;

    gm.reset();
    let cold_a = gm.prefill(&pb, &[a.clone(), ta.clone()].concat())?;
    gm.reset();
    let cold_b = gm.prefill(&pb, &[b.clone(), tb.clone()].concat())?;

    gm.reset();
    gm.restore_state(&arena, &parked_a)?;
    let reopened_a = gm.prefill(&pb, &ta)?;
    assert_close("A -> B -> A: A tail", &reopened_a, &cold_a);

    gm.reset();
    gm.restore_state(&arena, &parked_b)?;
    let reopened_b = gm.prefill(&pb, &tb)?;
    assert_close("A -> B -> B tail", &reopened_b, &cold_b);
    Ok(())
}
