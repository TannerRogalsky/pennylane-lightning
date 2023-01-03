use num_complex::Complex;
use std::simd::{Simd, SimdElement};

pub struct GateImplementationsAVXCommon;

impl crate::GateImplementation<f32> for GateImplementationsAVXCommon {
    const KERNEL_ID: crate::KernelType = crate::KernelType::AVX2;
    const ALIGNMENT: usize = std::mem::align_of::<f32>();
    const PACKED_BYTES: usize = 32;

    fn apply_pauli_x(
        data: &mut [num_complex::Complex<f32>],
        num_qubits: usize,
        wires: &[usize],
        adj: bool,
    ) {
        todo!()
    }

    fn apply_pauli_y(
        data: &mut [num_complex::Complex<f32>],
        num_qubits: usize,
        wires: &[usize],
        adj: bool,
    ) {
        todo!()
    }

    fn apply_pauli_z(
        data: &mut [num_complex::Complex<f32>],
        num_qubits: usize,
        wires: &[usize],
        adj: bool,
    ) {
        todo!()
    }
}

impl GateImplementationsAVXCommon {
    fn apply_pauli_x(data: &mut [Complex<f32>], num_qubits: usize, wires: &[usize], adj: bool) {
        assert_eq!(wires.len(), 1);
        const REV_WIRE: usize =
            <GateImplementationsAVXCommon as crate::GateImplementation<f32>>::PACKED_BYTES
                / std::mem::size_of::<f32>();

        ApplyPauliX::apply_internal(data, num_qubits, adj);
    }
}

struct ApplyPauliX;

impl ApplyPauliX {
    fn apply_internal(data: &mut [Complex<f32>], num_qubits: usize, _adj: bool) {
        const REV_WIRE: usize =
            <GateImplementationsAVXCommon as crate::GateImplementation<f32>>::PACKED_BYTES
                / std::mem::size_of::<f32>();

        const FLIPPED: [u8; 8] = flip(&identity(), REV_WIRE);
        // const FLIPPED: [u8; 8] = flip(&identity(), 0);
        const WITHIN_LANE: bool = is_within_lane::<f32, 8>(&FLIPPED);

        let compiled = if WITHIN_LANE {
            const IMM8: i32 = get_permutation_4x(&FLIPPED);
            CompiledPermutation::<_, IMM8>::WithinLane
        } else {
            CompiledPermutation::WithoutLane(get_permutation_8x256i(&FLIPPED))
        };
        for k in (0..(1 << num_qubits)).step_by(8 / 2) {
            let v = load(&data[k..]);
            let r = compiled.permute(v);
            store(r, &mut data[k..]);
        }
    }
}

fn load<F: SimdElement>(data: &[Complex<F>]) -> Simd<F, 8> {
    let data = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const F, data.len() * 2) };
    Simd::<F, 8>::from_slice(data)
}

fn store<F: SimdElement>(src: Simd<F, 8>, dst: &mut [Complex<F>]) {
    let src = unsafe {
        let data = src.as_array();
        std::slice::from_raw_parts(data.as_ptr() as *const Complex<F>, data.len() / 2)
    };
    dst.copy_from_slice(src);
}

const fn identity<const N: usize>() -> [u8; N] {
    let mut res = [0; N];

    let mut i = 0;
    while i < N {
        res[i] = i as _;
        i += 1;
    }

    res
}

const fn flip<const N: usize>(perm: &[u8; N], rev_wire: usize) -> [u8; N] {
    let mut res = [0; N];

    let mut k = 0;
    while k < N / 2 {
        res[2 * k + 0] = perm[2 * (k ^ (1 << rev_wire)) + 0];
        res[2 * k + 1] = perm[2 * (k ^ (1 << rev_wire)) + 1];
        k += 1;
    }

    res
}

pub const fn const_sort<const N: usize>(mut arr: [u8; N]) -> [u8; N] {
    loop {
        let mut swapped = false;
        let mut i = 1;
        while i < arr.len() {
            if arr[i - 1] > arr[i] {
                let left = arr[i - 1];
                let right = arr[i];
                arr[i - 1] = right;
                arr[i] = left;
                swapped = true;
            }
            i += 1;
        }
        if !swapped {
            break;
        }
    }
    arr
}

// const fn compile_permutation_f32(
//     permutation: &[u8; 8],
// ) -> CompiledPermutation<std::simd::i64x4, 8> {
//     const WITHIN_LANE: bool = is_within_lane::<f32, 8>(permutation);

//     if WITHIN_LANE {
//         const IMM8: i32 = get_permutation_4x(permutation);
//         CompiledPermutation::<_, IMM8>::WithinLane
//     } else {
//         CompiledPermutation::WithoutLane(get_permutation_8x256i(permutation))
//     }
// }

// enum PermutationData {
//     Float(),
//     Double(std::simd::i64x8),
// }

#[derive(Debug, Copy, Clone)]
pub enum CompiledPermutation<F, const IMM8: i32> {
    WithinLane,
    WithoutLane(F),
}

impl<const IMM8: i32> CompiledPermutation<std::simd::i64x4, IMM8> {
    pub const PACKED_SIZE: usize = 8;
    pub const IMM8: i32 = IMM8;

    fn permute(self, data: Simd<f32, 8>) -> Simd<f32, 8> {
        match self {
            CompiledPermutation::WithinLane => {
                let a = data.into();
                unsafe { std::arch::x86_64::_mm256_permute_ps(a, IMM8) }.into()
            }
            CompiledPermutation::WithoutLane(idx) => {
                let a = data.into();
                let idx = idx.into();
                unsafe { std::arch::x86_64::_mm256_permutevar8x32_ps(a, idx) }.into()
            }
        }
    }
}

trait LaneSize {
    const SIZE_WITHIN_LANE: usize;
}

impl<T> LaneSize for T {
    const SIZE_WITHIN_LANE: usize = 16 / std::mem::size_of::<Self>();
}

const fn is_within_lane<F: LaneSize, const N: usize>(permutation: &[u8; N]) -> bool {
    let mut lane = [0; N];

    let mut i = 0;
    while i < F::SIZE_WITHIN_LANE {
        lane[i] = permutation[i];
        i += 1;
    }

    {
        let lane2 = const_sort(lane);
        let mut i = 0;
        while i < F::SIZE_WITHIN_LANE {
            if lane2[i] != i as _ {
                return false;
            }
            i += 1;
        }
    }

    {
        let mut k = 0;
        while k < permutation.len() {
            let mut idx = 0;
            while idx < F::SIZE_WITHIN_LANE {
                if lane[idx] + (k as u8) != permutation[idx + k] {
                    return false;
                }
                idx += 1;
            }
            k += F::SIZE_WITHIN_LANE;
        }
    }

    true
}

const fn get_permutation_4x<const N: usize>(perm: &[u8; N]) -> i32 {
    let mut res = 0;
    let mut idx = 3;
    while idx > 0 {
        res <<= 2;
        res |= perm[idx] as i32;
        idx -= 1;
    }
    res
}

const fn get_permutation_8x256i(perm: &[u8; 8]) -> std::simd::i64x4 {
    setr256i(perm)
}

const fn setr256i(perm: &[u8; 8]) -> std::simd::i64x4 {
    std::simd::i64x4::from_array([
        ((perm[1] as i64) << 32) | perm[0] as i64,
        ((perm[3] as i64) << 32) | perm[2] as i64,
        ((perm[5] as i64) << 32) | perm[4] as i64,
        ((perm[7] as i64) << 32) | perm[6] as i64,
    ])
}
