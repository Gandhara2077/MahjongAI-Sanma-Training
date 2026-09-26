use crate::py_helper::add_submodule;

use pyo3::prelude::*;

// ============================================================================
// BUILD VARIANTS (yonma / sanma)
// ----------------------------------------------------------------------------
// This crate compiles exactly one rules variant per build, selected by the
// cargo feature `sanma`:
//
// - default (yonma): standard 4-player riichi mahjong; stays close to
//   upstream Equim-chan/Mortal.
// - `sanma`: 3-player rules (no chi, nukidora, tsumo-loss scoring, 108-tile
//   wall without 2m-8m, 35k starting score, hanchan of 6 kyoku, ...).
//
// OBS VERSION NAMESPACE -- DO NOT RENUMBER
// ----------------------------------------------------------------------------
// `obs_shape(version)` / `oracle_obs_shape(version)` are compile-time
// namespaced: version 4 means DIFFERENT layouts in the two builds.
//
// - yonma build: version 4 == (1012, 34), the native 4-player encoding.
// - sanma build: version 4 == (775, 34), the FROZEN MahjongCopilot /
//   libriichi3p deployment ABI (exported deployment models pin
//   `config.control.version = 4`); version 5 == (780, 34), the native sanma
//   encoding.
//
// Versions must NOT be renumbered: the (775, 34) layout at version 4 is an
// external deployment contract with MahjongCopilot. Within one compiled
// binary there is no ambiguity; ABI pin tests on both build legs guard this.
//
// ACTION SPACE LAYOUT
// ----------------------------------------------------------------------------
// - yonma (46): 0-36 discard/kan-choice, 37 riichi, 38-40 chi low/mid/high,
//   41 pon, 42 kan, 43 agari, 44 ryukyoku, 45 pass.
// - sanma (44): 0-36 discard/kan-choice, 37 riichi, 38 pon, 39 kan,
//   40 nukidora, 41 agari, 42 ryukyoku, 43 pass. Chi is disabled in sanma;
//   the reused slots 38=pon / 39=kan / 40=nukidora are the frozen
//   MahjongCopilot deployment ABI and must not change.
// ============================================================================

/// Number of players: 4 for yonma (default), 3 for sanma.
#[cfg(not(feature = "sanma"))]
pub const NUM_PLAYERS: usize = 4;
/// Number of players: 4 for yonma (default), 3 for sanma.
#[cfg(feature = "sanma")]
pub const NUM_PLAYERS: usize = 3;

#[cfg(not(feature = "sanma"))]
pub const MAX_VERSION: u32 = 4;
#[cfg(feature = "sanma")]
pub const MAX_VERSION: u32 = 5;

pub const ACTION_RIICHI: usize = 37;
#[cfg(not(feature = "sanma"))]
pub const ACTION_CHI_LOW: usize = 38;
#[cfg(not(feature = "sanma"))]
pub const ACTION_CHI_MID: usize = 39;
#[cfg(not(feature = "sanma"))]
pub const ACTION_CHI_HIGH: usize = 40;
#[cfg(not(feature = "sanma"))]
pub const ACTION_PON: usize = 41;
#[cfg(not(feature = "sanma"))]
pub const ACTION_KAN: usize = 42;
#[cfg(feature = "sanma")]
pub const ACTION_PON: usize = 38;
#[cfg(feature = "sanma")]
pub const ACTION_KAN: usize = 39;
#[cfg(feature = "sanma")]
pub const ACTION_NUKIDORA: usize = 40;
#[cfg(not(feature = "sanma"))]
pub const ACTION_AGARI: usize = 43;
#[cfg(feature = "sanma")]
pub const ACTION_AGARI: usize = 41;
#[cfg(not(feature = "sanma"))]
pub const ACTION_RYUKYOKU: usize = 44;
#[cfg(feature = "sanma")]
pub const ACTION_RYUKYOKU: usize = 42;
#[cfg(not(feature = "sanma"))]
pub const ACTION_PASS: usize = 45;
#[cfg(feature = "sanma")]
pub const ACTION_PASS: usize = 43;
pub const ACTION_SPACE: usize = ACTION_PASS + 1;

/// GRP input size: [grand_kyoku, honba, kyotaku, s0..s(NUM_PLAYERS-1)].
#[cfg(not(feature = "sanma"))]
pub const GRP_SIZE: usize = 7;
/// GRP input size: [grand_kyoku, honba, kyotaku, s0..s(NUM_PLAYERS-1)].
#[cfg(feature = "sanma")]
pub const GRP_SIZE: usize = 6;

/// Live-wall tiles at kyoku start: 136 - 13*4 haipai - 14 wanpai = 70 for
/// yonma; 108 - 13*3 - 14 = 55 for sanma (no 2m-8m in the wall).
#[cfg(not(feature = "sanma"))]
pub(crate) const INITIAL_TILES_LEFT: u8 = 70;
/// Live-wall tiles at kyoku start: 136 - 13*4 haipai - 14 wanpai = 70 for
/// yonma; 108 - 13*3 - 14 = 55 for sanma (no 2m-8m in the wall).
#[cfg(feature = "sanma")]
pub(crate) const INITIAL_TILES_LEFT: u8 = 55;

/// Minimum live-wall tiles required to declare riichi: 4 for yonma. Sanma
/// uses 3: with three players, three live-wall tiles still guarantee one
/// more draw for the declarer.
#[cfg(not(feature = "sanma"))]
pub(crate) const MIN_TILES_LEFT_FOR_RIICHI: u8 = 4;
/// Minimum live-wall tiles required to declare riichi: 4 for yonma. Sanma
/// uses 3: with three players, three live-wall tiles still guarantee one
/// more draw for the declarer.
#[cfg(feature = "sanma")]
pub(crate) const MIN_TILES_LEFT_FOR_RIICHI: u8 = 3;

/// Honba surcharge each payer pays to a tsumo winner: 100 in yonma
/// (3 payers × 100 = 300 to the winner), 150 in sanma (2 payers × 150 = 300
/// to the winner). Used by both the arena score settlement and the agent
/// helper's post-hora estimation.
#[cfg(not(feature = "sanma"))]
pub(crate) const HONBA_TSUMO_SURCHARGE: i32 = 100;
/// Honba surcharge each payer pays to a tsumo winner: 100 in yonma
/// (3 payers × 100 = 300 to the winner), 150 in sanma (2 payers × 150 = 300
/// to the winner). Used by both the arena score settlement and the agent
/// helper's post-hora estimation.
#[cfg(feature = "sanma")]
pub(crate) const HONBA_TSUMO_SURCHARGE: i32 = 150;

/// Starting score per player: 25,000 in yonma (100k table), 35,000 in
/// sanma (105k table).
#[cfg(not(feature = "sanma"))]
pub(crate) const STARTING_SCORE: i32 = 25000;
/// Starting score per player: 25,000 in yonma (100k table), 35,000 in
/// sanma (105k table).
#[cfg(feature = "sanma")]
pub(crate) const STARTING_SCORE: i32 = 35000;

#[pyfunction]
#[inline]
pub const fn obs_shape(version: u32) -> (usize, usize) {
    match version {
        1 => (938, 34),
        2 => (942, 34),
        3 => (934, 34),
        // yonma: native 4-player encoding.
        #[cfg(not(feature = "sanma"))]
        4 => (1012, 34),
        // sanma: MahjongCopilot/libriichi3p deployment ABI (frozen, see the
        // banner at the top of this file).
        #[cfg(feature = "sanma")]
        4 => (775, 34),
        // sanma: native sanma encoding.
        #[cfg(feature = "sanma")]
        5 => (780, 34),
        _ => unreachable!(),
    }
}

#[pyfunction]
#[inline]
pub const fn oracle_obs_shape(version: u32) -> (usize, usize) {
    match version {
        1 => (211, 34),
        2 | 3 | 4 => (217, 34),
        #[cfg(feature = "sanma")]
        5 => (170, 34),
        _ => unreachable!(),
    }
}

pub(crate) fn register_module(
    py: Python<'_>,
    prefix: &str,
    super_mod: &Bound<'_, PyModule>,
) -> PyResult<()> {
    let m = PyModule::new(py, "consts")?;
    m.add_function(wrap_pyfunction!(obs_shape, &m)?)?;
    m.add_function(wrap_pyfunction!(oracle_obs_shape, &m)?)?;
    m.add("MAX_VERSION", MAX_VERSION)?;
    m.add("ACTION_SPACE", ACTION_SPACE)?;
    m.add("ACTION_RIICHI", ACTION_RIICHI)?;
    #[cfg(not(feature = "sanma"))]
    {
        m.add("ACTION_CHI_LOW", ACTION_CHI_LOW)?;
        m.add("ACTION_CHI_MID", ACTION_CHI_MID)?;
        m.add("ACTION_CHI_HIGH", ACTION_CHI_HIGH)?;
    }
    m.add("ACTION_PON", ACTION_PON)?;
    m.add("ACTION_KAN", ACTION_KAN)?;
    #[cfg(feature = "sanma")]
    m.add("ACTION_NUKIDORA", ACTION_NUKIDORA)?;
    m.add("ACTION_AGARI", ACTION_AGARI)?;
    m.add("ACTION_RYUKYOKU", ACTION_RYUKYOKU)?;
    m.add("ACTION_PASS", ACTION_PASS)?;
    m.add("GRP_SIZE", GRP_SIZE)?;
    m.add("NUM_PLAYERS", NUM_PLAYERS)?;
    add_submodule(py, prefix, super_mod, &m)
}

/// Pins the yonma (default build) deployment ABI. The sanma counterpart
/// lives in `sanma_compat_tests.rs` (only compiled with `feature =
/// "sanma"`).
#[cfg(all(test, not(feature = "sanma")))]
mod tests {
    use super::*;

    #[test]
    fn yonma_abi_is_stable() {
        assert_eq!(NUM_PLAYERS, 4);
        assert_eq!(MAX_VERSION, 4);
        assert_eq!(ACTION_SPACE, 46);
        assert_eq!(GRP_SIZE, 7);
        assert_eq!(obs_shape(4), (1012, 34));
        assert_eq!(oracle_obs_shape(4), (217, 34));
        assert_eq!(ACTION_RIICHI, 37);
        assert_eq!(ACTION_CHI_LOW, 38);
        assert_eq!(ACTION_CHI_MID, 39);
        assert_eq!(ACTION_CHI_HIGH, 40);
        assert_eq!(ACTION_PON, 41);
        assert_eq!(ACTION_KAN, 42);
        assert_eq!(ACTION_AGARI, 43);
        assert_eq!(ACTION_RYUKYOKU, 44);
        assert_eq!(ACTION_PASS, 45);
    }
}
