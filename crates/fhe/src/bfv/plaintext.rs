//! Plaintext type in the BFV encryption scheme.
use crate::{
	bfv::{BfvParameters, Encoding, PlaintextVec},
	Error, Result,
};
use fhe_traits::{FheDecoder, FheEncoder, FheParametrized, FhePlaintext};
use math::rq::{traits::TryConvertFrom, Context, Poly, Representation};
use std::sync::Arc;
use zeroize::{Zeroize, ZeroizeOnDrop, Zeroizing};

use super::encoding::EncodingEnum;

/// A plaintext object, that encodes a vector according to a specific encoding.
#[derive(Debug, Clone, Eq)]
pub struct Plaintext {
	/// The parameters of the underlying BFV encryption scheme.
	pub(crate) par: Arc<BfvParameters>,
	/// The value after encoding.
	pub(crate) value: Box<[u64]>,
	/// The encoding of the plaintext, if known
	pub(crate) encoding: Option<Encoding>,
	/// The plaintext as a polynomial.
	pub(crate) poly_ntt: Poly,
	/// The level of the plaintext
	pub(crate) level: usize,
}

impl FheParametrized for Plaintext {
	type Parameters = BfvParameters;
}

impl FhePlaintext for Plaintext {
	type Encoding = Encoding;
}

// Zeroizing of plaintexts.
impl ZeroizeOnDrop for Plaintext {}

impl Zeroize for Plaintext {
	fn zeroize(&mut self) {
		self.value.zeroize();
		self.poly_ntt.zeroize();
	}
}

impl Plaintext {
	pub(crate) fn to_poly(&self) -> Result<Poly> {
		let mut m_v = Zeroizing::new(self.value.clone());
		self.par
			.plaintext
			.scalar_mul_vec(&mut m_v, self.par.q_mod_t[self.level]);
		let ctx = self.par.ctx_at_level(self.level)?;
		let mut m = Poly::try_convert_from(m_v.as_ref(), ctx, false, Representation::PowerBasis)?;
		m.change_representation(Representation::Ntt);
		m *= &self.par.delta[self.level];
		Ok(m)
	}

	/// Generate a zero plaintext.
	pub fn zero(encoding: Encoding, par: &Arc<BfvParameters>) -> Result<Self> {
		let level = encoding.level;
		let ctx = par.ctx_at_level(level)?;
		let value = vec![0u64; par.degree()];
		let poly_ntt = Poly::zero(ctx, Representation::Ntt);
		Ok(Self {
			par: par.clone(),
			value: value.into_boxed_slice(),
			encoding: Some(encoding),
			poly_ntt,
			level,
		})
	}

	/// Returns the level of this plaintext.
	pub fn level(&self) -> usize {
		self.par.level_of_ctx(self.poly_ntt.ctx()).unwrap()
	}
}

unsafe impl Send for Plaintext {}

// Implement the equality manually; we want to say that two plaintexts are equal
// even if one of them doesn't store its encoding information.
impl PartialEq for Plaintext {
	fn eq(&self, other: &Self) -> bool {
		let mut eq = self.par == other.par;
		eq &= self.value == other.value;
		if self.encoding.is_some() && other.encoding.is_some() {
			eq &= self.encoding.as_ref().unwrap() == other.encoding.as_ref().unwrap()
		}
		eq
	}
}

// Conversions.
impl TryConvertFrom<&Plaintext> for Poly {
	fn try_convert_from<R>(
		pt: &Plaintext,
		ctx: &Arc<Context>,
		variable_time: bool,
		_: R,
	) -> math::Result<Self>
	where
		R: Into<Option<Representation>>,
	{
		if ctx
			!= pt
				.par
				.ctx_at_level(pt.level())
				.map_err(|e| math::Error::Default(e.to_string()))?
		{
			Err(math::Error::Default("Incompatible contexts".to_string()))
		} else {
			Poly::try_convert_from(
				&pt.value as &[u64],
				ctx,
				variable_time,
				Representation::PowerBasis,
			)
		}
	}
}

// Encoding and decoding.

impl FheEncoder<&[u64]> for Plaintext {
	type Error = Error;
	fn try_encode(value: &[u64], encoding: Encoding, par: &Arc<BfvParameters>) -> Result<Self> {
		if value.len() > par.degree() {
			return Err(Error::TooManyValues(value.len(), par.degree()));
		}
		let v = PlaintextVec::try_encode(value, encoding, par)?;
		Ok(v.0[0].clone())
	}
}

impl FheEncoder<&[i64]> for Plaintext {
	type Error = Error;
	fn try_encode(value: &[i64], encoding: Encoding, par: &Arc<BfvParameters>) -> Result<Self> {
		let w = Zeroizing::new(par.plaintext.reduce_vec_i64(value));
		Plaintext::try_encode(&w as &[u64], encoding, par)
	}
}

impl FheDecoder<Plaintext> for Vec<u64> {
	fn try_decode<O>(pt: &Plaintext, encoding: O) -> Result<Vec<u64>>
	where
		O: Into<Option<Encoding>>,
	{
		let encoding = encoding.into();
		let enc: Encoding;
		if pt.encoding.is_none() && encoding.is_none() {
			return Err(Error::UnspecifiedInput("No encoding specified".to_string()));
		} else if pt.encoding.is_some() {
			enc = pt.encoding.as_ref().unwrap().clone();
			if let Some(arg_enc) = encoding && arg_enc != enc {
				return Err(Error::EncodingMismatch(arg_enc.into(), enc.into()))
			}
		} else {
			enc = encoding.unwrap();
			if let Some(pt_enc) = pt.encoding.as_ref() && pt_enc != &enc {
				return Err(Error::EncodingMismatch(pt_enc.into(), enc.into()))
			}
		}

		let mut w = pt.value.to_vec();

		match enc.encoding {
			EncodingEnum::Poly => Ok(w),
			EncodingEnum::Simd => {
				if let Some(op) = &pt.par.op {
					op.forward(&mut w);
					let mut w_reordered = w.clone();
					for i in 0..pt.par.degree() {
						w_reordered[i] = w[pt.par.matrix_reps_index_map[i]]
					}
					w.zeroize();
					Ok(w_reordered)
				} else {
					Err(Error::EncodingNotSupported(EncodingEnum::Simd.to_string()))
				}
			}
		}
	}

	type Error = Error;
}

impl FheDecoder<Plaintext> for Vec<i64> {
	fn try_decode<E>(pt: &Plaintext, encoding: E) -> Result<Vec<i64>>
	where
		E: Into<Option<Encoding>>,
	{
		let v = Vec::<u64>::try_decode(pt, encoding)?;
		Ok(unsafe { pt.par.plaintext.center_vec_vt(&v) })
	}

	type Error = Error;
}

#[cfg(test)]
mod tests {
	use super::{Encoding, Plaintext};
	use crate::bfv::parameters::{BfvParameters, BfvParametersBuilder};
	use fhe_traits::{FheDecoder, FheEncoder};
	use math::rq::{Poly, Representation};
	use std::{error::Error, sync::Arc};
	use zeroize::Zeroize;

	#[test]
	fn try_encode() -> Result<(), Box<dyn Error>> {
		// The default test parameters support both Poly and Simd encodings
		let params = Arc::new(BfvParameters::default(1, 8));
		let a = params.plaintext.random_vec(params.degree());

		let plaintext = Plaintext::try_encode(&[0u64; 9] as &[u64], Encoding::poly(), &params);
		assert!(plaintext.is_err());

		let plaintext = Plaintext::try_encode(&a as &[u64], Encoding::poly(), &params);
		assert!(plaintext.is_ok());

		let plaintext = Plaintext::try_encode(&a as &[u64], Encoding::simd(), &params);
		assert!(plaintext.is_ok());

		let plaintext = Plaintext::try_encode(&[1u64] as &[u64], Encoding::poly(), &params);
		assert!(plaintext.is_ok());

		// The following parameters do not allow for Simd encoding
		let params = Arc::new(
			BfvParametersBuilder::new()
				.set_degree(8)
				.set_plaintext_modulus(2)
				.set_moduli(&[4611686018326724609])
				.build()?,
		);

		let a = params.plaintext.random_vec(params.degree());

		let plaintext = Plaintext::try_encode(&a as &[u64], Encoding::poly(), &params);
		assert!(plaintext.is_ok());

		let plaintext = Plaintext::try_encode(&a as &[u64], Encoding::simd(), &params);
		assert!(plaintext.is_err());

		Ok(())
	}

	#[test]
	fn encode_decode() {
		(0..40).for_each(|_| {
			let params = Arc::new(BfvParameters::default(1, 8));
			let a = params.plaintext.random_vec(params.degree());

			let plaintext = Plaintext::try_encode(&a as &[u64], Encoding::simd(), &params);
			assert!(plaintext.is_ok());
			let b = Vec::<u64>::try_decode(&plaintext.unwrap(), Encoding::simd());
			assert!(b.is_ok_and(|b| b == &a));

			let a = unsafe { params.plaintext.center_vec_vt(&a) };
			let plaintext = Plaintext::try_encode(&a as &[i64], Encoding::poly(), &params);
			assert!(plaintext.is_ok());
			let b = Vec::<i64>::try_decode(&plaintext.unwrap(), Encoding::poly());
			assert!(b.is_ok_and(|b| b == &a));

			let plaintext = Plaintext::try_encode(&a as &[i64], Encoding::simd(), &params);
			assert!(plaintext.is_ok());
			let b = Vec::<i64>::try_decode(&plaintext.unwrap(), Encoding::simd());
			assert!(b.is_ok_and(|b| b == &a));
		})
	}

	#[test]
	fn partial_eq() -> Result<(), Box<dyn Error>> {
		let params = Arc::new(BfvParameters::default(1, 8));
		let a = params.plaintext.random_vec(params.degree());

		let plaintext = Plaintext::try_encode(&a as &[u64], Encoding::poly(), &params)?;
		let mut same_plaintext = Plaintext::try_encode(&a as &[u64], Encoding::poly(), &params)?;
		assert_eq!(plaintext, same_plaintext);

		// Equality also holds when there is no encoding specified. In this test, we use
		// the fact that we can set it to None directly, but such a partial plaintext
		// will be created during decryption since we do not specify the encoding at the
		// time.
		same_plaintext.encoding = None;
		assert_eq!(plaintext, same_plaintext);

		Ok(())
	}

	#[test]
	fn try_decode_errors() -> Result<(), Box<dyn Error>> {
		let params = Arc::new(BfvParameters::default(1, 8));
		let a = params.plaintext.random_vec(params.degree());

		let mut plaintext = Plaintext::try_encode(&a as &[u64], Encoding::poly(), &params)?;

		assert!(Vec::<u64>::try_decode(&plaintext, None).is_ok());
		assert!(
			Vec::<u64>::try_decode(&plaintext, Encoding::simd()).is_err_and(|err| err
				== &crate::Error::EncodingMismatch(
					Encoding::simd().into(),
					Encoding::poly().into()
				))
		);
		assert!(
			Vec::<u64>::try_decode(&plaintext, Encoding::poly_at_level(1)).is_err_and(|err| err
				== &crate::Error::EncodingMismatch(
					Encoding::poly_at_level(1).into(),
					Encoding::poly().into()
				))
		);

		plaintext.encoding = None;
		assert!(Vec::<u64>::try_decode(&plaintext, None).is_err_and(
			|err| err == &crate::Error::UnspecifiedInput("No encoding specified".to_string())
		));

		Ok(())
	}

	#[test]
	fn zero() -> Result<(), Box<dyn Error>> {
		let params = Arc::new(BfvParameters::default(1, 8));
		let plaintext = Plaintext::zero(Encoding::poly(), &params)?;

		assert_eq!(plaintext.value, Box::<[u64]>::from([0u64; 8]));
		assert_eq!(
			plaintext.poly_ntt,
			Poly::zero(&params.ctx[0], Representation::Ntt)
		);

		Ok(())
	}

	#[test]
	fn zeroize() -> Result<(), Box<dyn Error>> {
		let params = Arc::new(BfvParameters::default(1, 8));
		let a = params.plaintext.random_vec(params.degree());
		let mut plaintext = Plaintext::try_encode(&a as &[u64], Encoding::poly(), &params)?;

		plaintext.zeroize();

		assert_eq!(plaintext, Plaintext::zero(Encoding::poly(), &params)?);

		Ok(())
	}

	#[test]
	fn try_encode_level() -> Result<(), Box<dyn Error>> {
		// The default test parameters support both Poly and Simd encodings
		let params = Arc::new(BfvParameters::default(10, 8));
		let a = params.plaintext.random_vec(params.degree());

		for level in 0..10 {
			let plaintext =
				Plaintext::try_encode(&a as &[u64], Encoding::poly_at_level(level), &params)?;
			assert_eq!(plaintext.level(), level);
			let plaintext =
				Plaintext::try_encode(&a as &[u64], Encoding::simd_at_level(level), &params)?;
			assert_eq!(plaintext.level(), level);
		}

		Ok(())
	}
}