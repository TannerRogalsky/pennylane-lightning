pub const fn fill_trailing_ones(nbits: usize) -> usize {
    if nbits == 0 {
        0
    } else {
        !0 >> (u8::BITS as usize * std::mem::size_of::<usize>() - nbits)
    }
}

pub const fn fill_leading_ones(nbits: usize) -> usize {
    !0 << nbits
}

pub const fn exp2(n: usize) -> usize {
    1 << n
}
