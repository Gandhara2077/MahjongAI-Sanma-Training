//! Sanma build ABI pins and wire-format rejection tests.
//!
//! These tests only compile under `#[cfg(all(test, feature = "sanma"))]`
//! (see `lib.rs`). The yonma counterpart of the ABI pin lives in
//! `consts::tests::yonma_abi_is_stable`.
use crate::consts::{
    ACTION_AGARI, ACTION_KAN, ACTION_NUKIDORA, ACTION_PASS, ACTION_PON, ACTION_RIICHI,
    ACTION_RYUKYOKU, ACTION_SPACE, GRP_SIZE, MAX_VERSION, NUM_PLAYERS, obs_shape, oracle_obs_shape,
};
use crate::mjai::Event;

/// Pins the MahjongCopilot/libriichi3p deployment contract: obs v4 layout
/// (775, 34) and action slots 38=pon / 39=kan / 40=nukidora. Exported
/// deployment models depend on these exact values; see the banner in
/// `consts.rs`.
#[test]
fn mahjongcopilot_v4_abi_is_stable() {
    assert_eq!(NUM_PLAYERS, 3);
    assert_eq!(MAX_VERSION, 5);
    assert_eq!(ACTION_SPACE, 44);
    assert_eq!(GRP_SIZE, 6);
    assert_eq!(obs_shape(4), (775, 34));
    assert_eq!(obs_shape(5), (780, 34));
    assert_eq!(oracle_obs_shape(4), (217, 34));
    assert_eq!(oracle_obs_shape(5), (170, 34));
    assert_eq!(ACTION_RIICHI, 37);
    assert_eq!(ACTION_PON, 38);
    assert_eq!(ACTION_KAN, 39);
    assert_eq!(ACTION_NUKIDORA, 40);
    assert_eq!(ACTION_AGARI, 41);
    assert_eq!(ACTION_RYUKYOKU, 42);
    assert_eq!(ACTION_PASS, 43);
}

#[test]
fn nukidora_event_round_trips() {
    let raw = r#"{"type":"nukidora","actor":1,"pai":"N"}"#;
    let event: Event = serde_json::from_str(raw).expect("nukidora must be accepted");
    assert_eq!(event.actor(), Some(1));
    assert_eq!(serde_json::to_string(&event).unwrap(), raw);
}

#[test]
fn fourth_actor_is_rejected_in_sanma() {
    let raw = r#"{"type":"tsumo","actor":3,"pai":"1p"}"#;
    serde_json::from_str::<Event>(raw).unwrap_err();
}

#[test]
fn east_four_is_rejected_in_sanma() {
    let raw = r#"{
        "type":"start_kyoku",
        "bakaze":"E",
        "dora_marker":"1p",
        "kyoku":4,
        "honba":0,
        "kyotaku":0,
        "oya":0,
        "scores":[35000,35000,35000,0],
        "tehais":[
            ["1m","1m","9m","9m","1p","2p","3p","4p","5p","6p","7p","8p","9p"],
            ["?","?","?","?","?","?","?","?","?","?","?","?","?"],
            ["?","?","?","?","?","?","?","?","?","?","?","?","?"],
            ["?","?","?","?","?","?","?","?","?","?","?","?","?"]
        ]
    }"#;
    serde_json::from_str::<Event>(raw).unwrap_err();
}
