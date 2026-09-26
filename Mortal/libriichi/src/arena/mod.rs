mod board;
mod game;
#[cfg(not(feature = "sanma"))]
mod one_vs_three;
#[cfg(feature = "sanma")]
mod one_vs_two;
mod result;
#[cfg(feature = "sanma")]
mod three_way;
#[cfg(not(feature = "sanma"))]
mod two_vs_two;

pub use board::Board;
pub use result::GameResult;

use crate::py_helper::add_submodule;
#[cfg(not(feature = "sanma"))]
use one_vs_three::OneVsThree;
#[cfg(feature = "sanma")]
use one_vs_two::OneVsTwo;
#[cfg(feature = "sanma")]
use three_way::ThreeWay;
#[cfg(not(feature = "sanma"))]
use two_vs_two::TwoVsTwo;

use pyo3::prelude::*;

pub(crate) fn register_module(
    py: Python<'_>,
    prefix: &str,
    super_mod: &Bound<'_, PyModule>,
) -> PyResult<()> {
    let m = PyModule::new(py, "arena")?;
    // Yonma arena modes (OneVsThree incl. akochan, TwoVsTwo) only build in
    // the default variant; sanma builds get OneVsTwo and ThreeWay instead.
    #[cfg(not(feature = "sanma"))]
    {
        m.add_class::<OneVsThree>()?;
        m.add_class::<TwoVsTwo>()?;
    }
    #[cfg(feature = "sanma")]
    {
        m.add_class::<OneVsTwo>()?;
        m.add_class::<ThreeWay>()?;
    }
    add_submodule(py, prefix, super_mod, &m)
}
