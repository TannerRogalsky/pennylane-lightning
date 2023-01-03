#![feature(portable_simd)]

mod gate_implementations_avx_common;
mod gate_implementations_lm;
mod utils;

pub use gate_implementations_avx_common::GateImplementationsAVXCommon;
pub use gate_implementations_lm::GateImplementationsLM;

pub enum KernelType {
    LM,
    PI,
    AVX2,
    AVX512,
}

pub trait GateImplementation<F: num_traits::Float + num_traits::FloatConst> {
    const KERNEL_ID: KernelType;
    const ALIGNMENT: usize;
    const PACKED_BYTES: usize;

    fn apply_pauli_x(
        data: &mut [num_complex::Complex<F>],
        num_qubits: usize,
        wires: &[usize],
        adj: bool,
    );
    fn apply_pauli_y(
        data: &mut [num_complex::Complex<F>],
        num_qubits: usize,
        wires: &[usize],
        adj: bool,
    );
    fn apply_pauli_z(
        data: &mut [num_complex::Complex<F>],
        num_qubits: usize,
        wires: &[usize],
        adj: bool,
    );
}
