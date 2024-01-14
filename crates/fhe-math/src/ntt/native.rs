use crate::zq::Modulus;

use concrete_ntt::prime64::Plan;
/// Number-Theoretic Transform operator.
#[derive(Debug, Clone)]
pub struct NttOperator {
    p: Modulus,
    size: usize,
    ntt_plan: Plan,
}

impl PartialEq for NttOperator {
    fn eq(&self, oth: &Self) -> bool {
        self.p == oth.p && self.size == oth.size
    }
}

impl Eq for NttOperator {}

impl NttOperator {
    /// Create an NTT operator given a modulus for a specific size.
    ///
    /// Aborts if the size is not a power of 2 that is >= 8 in debug mode.
    /// Returns None if the modulus does not support the NTT for this specific
    /// size.
    pub fn new(p: &Modulus, size: usize) -> Option<Self> {
        if !super::supports_ntt(p.p, size) {
            return None;
        }

        let some_ntt_plan = Plan::try_new(size, p.p);
        if let Some(ntt_plan) = some_ntt_plan {
            Some(Self {
                p: p.clone(),
                size,
                ntt_plan,
            })
        } else {
            None
        }
    }

    /// Compute the forward NTT in place.
    /// Aborts if a is not of the size handled by the operator.
    pub fn forward(&self, a: &mut [u64]) {
        debug_assert_eq!(a.len(), self.size);
        self.ntt_plan.fwd(a);
    }

    /// Compute the backward NTT in place.
    /// Aborts if a is not of the size handled by the operator.
    pub fn backward(&self, a: &mut [u64]) {
        debug_assert_eq!(a.len(), self.size);
        self.ntt_plan.inv(a);
        self.ntt_plan.normalize(a);
    }

    /// Compute the forward NTT in place in variable time in a lazily fashion.
    /// This means that the output coefficients may be up to 4 times the
    /// modulus.
    ///
    /// # Safety
    /// This function assumes that a_ptr points to at least `size` elements.
    /// This function is not constant time and its timing may reveal information
    /// about the value being reduced.
    pub(crate) unsafe fn forward_vt_lazy(&self, a_ptr: *mut u64) {
        unsafe {
            let a = std::slice::from_raw_parts_mut(a_ptr, self.size);
            self.ntt_plan.fwd(a);
        }
    }

    /// Compute the forward NTT in place in variable time.
    ///
    /// # Safety
    /// This function assumes that a_ptr points to at least `size` elements.
    /// This function is not constant time and its timing may reveal information
    /// about the value being reduced.
    pub unsafe fn forward_vt(&self, a_ptr: *mut u64) {
        self.forward_vt_lazy(a_ptr);
        for i in 0..self.size {
            *a_ptr.add(i) = self.reduce3_vt(*a_ptr.add(i))
        }
    }

    /// Compute the backward NTT in place in variable time.
    ///
    /// # Safety
    /// This function assumes that a_ptr points to at least `size` elements.
    /// This function is not constant time and its timing may reveal information
    /// about the value being reduced.
    pub unsafe fn backward_vt(&self, a_ptr: *mut u64) {
        unsafe {
            let a = std::slice::from_raw_parts_mut(a_ptr, self.size);
            self.ntt_plan.inv(a);
            self.ntt_plan.normalize(a);
        }
    }

    /// Reduce a modulo p in variable time.
    ///
    /// Aborts if a >= 4 * p.
    const unsafe fn reduce3_vt(&self, a: u64) -> u64 {
        debug_assert!(a < 4 * self.p.p);

        let y = Modulus::reduce1_vt(a, 2 * self.p.p);
        Modulus::reduce1_vt(y, self.p.p)
    }
}
