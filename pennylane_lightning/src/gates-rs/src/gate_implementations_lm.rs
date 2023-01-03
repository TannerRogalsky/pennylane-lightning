use crate::{utils, GateImplementation, KernelType};
use num_complex::Complex;

pub struct GateImplementationsLM;

impl GateImplementationsLM {
    const fn rev_wire_parity(rev_wire: usize) -> (usize, usize) {
        let parity_low = utils::fill_trailing_ones(rev_wire);
        let parity_high = utils::fill_leading_ones(rev_wire + 1);

        (parity_high, parity_low)
    }

    fn apply_pauli_x<F>(data: &mut [Complex<F>], num_qubits: usize, wires: &[usize], _adj: bool)
    where
        F: num_traits::Float + num_traits::FloatConst,
    {
        assert_eq!(wires.len(), 1);

        let rev_wire = num_qubits - wires[0] - 1;
        let rev_wire_shift = 1 << rev_wire;

        let (parity_high, parity_low) = Self::rev_wire_parity(rev_wire);
        for k in 0..utils::exp2(num_qubits - 1) {
            let i0 = ((k << 1) & parity_high) | (parity_low & k);
            let i1 = i0 | rev_wire_shift;
            data.swap(i0, i1);
        }
    }

    fn apply_pauli_y<F>(data: &mut [Complex<F>], num_qubits: usize, wires: &[usize], _adj: bool)
    where
        F: num_traits::Float + num_traits::FloatConst,
    {
        assert_eq!(wires.len(), 1);

        let rev_wire = num_qubits - wires[0] - 1;
        let rev_wire_shift = 1 << rev_wire;

        let (parity_high, parity_low) = Self::rev_wire_parity(rev_wire);
        for k in 0..utils::exp2(num_qubits - 1) {
            let i0 = ((k << 1) & parity_high) | (parity_low & k);
            let i1 = i0 | rev_wire_shift;

            let v0 = data[i0];
            let v1 = data[i1];

            data[i0] = Complex::new(v1.im, -v1.re);
            data[i1] = Complex::new(-v0.im, v0.re);
        }
    }

    fn apply_pauli_z<F>(data: &mut [Complex<F>], num_qubits: usize, wires: &[usize], _adj: bool)
    where
        F: num_traits::Float + num_traits::FloatConst,
    {
        assert_eq!(wires.len(), 1);

        let rev_wire = num_qubits - wires[0] - 1;
        let rev_wire_shift = 1 << rev_wire;

        let (parity_high, parity_low) = Self::rev_wire_parity(rev_wire);
        let scale = -<F as num_traits::One>::one();
        for k in 0..utils::exp2(num_qubits - 1) {
            let i0 = ((k << 1) & parity_high) | (parity_low & k);
            let i1 = i0 | rev_wire_shift;
            data[i1].scale(scale);
        }
    }
}

impl<F> GateImplementation<F> for GateImplementationsLM
where
    F: num_traits::Float + num_traits::FloatConst,
{
    const KERNEL_ID: KernelType = KernelType::LM;
    const ALIGNMENT: usize = std::mem::align_of::<F>();
    const PACKED_BYTES: usize = std::mem::size_of::<F>();

    fn apply_pauli_x(data: &mut [Complex<F>], num_qubits: usize, wires: &[usize], adj: bool) {
        Self::apply_pauli_x(data, num_qubits, wires, adj)
    }

    fn apply_pauli_y(data: &mut [Complex<F>], num_qubits: usize, wires: &[usize], adj: bool) {
        Self::apply_pauli_y(data, num_qubits, wires, adj)
    }

    fn apply_pauli_z(data: &mut [Complex<F>], num_qubits: usize, wires: &[usize], adj: bool) {
        Self::apply_pauli_z(data, num_qubits, wires, adj)
    }
}
