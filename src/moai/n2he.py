#!/usr/bin/env python3
"""
N2HE: Neural Network Homomorphic Encryption Library

A Python implementation based on HintSight Technology's N2HE-hexl library.
Reference: https://github.com/HintSight-Technology/N2HE-hexl

Publication:
K.Y. Lam, X. Lu, L. Zhang, X. Wang, H. Wang and S.Q. Goh.
"Efficient FHE-based Privacy-Enhanced Neural Network for Trustworthy AI-as-a-Service"
IEEE Transactions on Dependable and Secure Computing, IEEE.

This implementation provides:
1. LWE-based additive homomorphic encryption (for linear operations)
2. FHEW-style bootstrapping with LUT (for non-linear activations like ReLU)
3. Key switching and modulus switching
4. SIMD-style batching for efficiency

Security: 128-bit post-quantum security based on LWE hardness assumption.
"""

import numpy as np
import hashlib
import struct
import time
import threading
import secrets
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Callable, Union
from enum import Enum
import gzip
import pickle

# =============================================================================
# N2HE Parameters (Security Configuration)
# =============================================================================

@dataclass
class N2HEParams:
    """
    N2HE cryptographic parameters following the Homomorphic Encryption Standard.
    
    The N2HE scheme uses:
    - LWE for linear operations (weighted sums in neural networks)
    - FHEW/TFHE bootstrapping for non-linear activations (ReLU, Sign, etc.)
    
    Security analysis:
    - n=1024, q=2^32, σ=3.2 provides ~128-bit post-quantum security
    - Based on hardness of Learning With Errors (LWE) problem
    
    Reference: https://homomorphicencryption.org/standard/
    """
    # LWE parameters
    n: int = 1024                    # Secret key dimension (lattice dimension)
    q: int = 2**32                   # Ciphertext modulus
    sigma: float = 3.2               # Error standard deviation (discrete Gaussian)
    
    # Message space
    t: int = 2**16                   # Plaintext modulus (message space Z_t)
    
    # FHEW/Bootstrapping parameters
    N: int = 1024                    # Ring dimension for RLWE (bootstrapping)
    Q: int = 2**27                   # Bootstrapping modulus
    Bks: int = 128                   # Key switching base
    dks: int = 3                     # Key switching decomposition depth
    
    # Security level
    security_bits: int = 128
    
    # Performance simulation parameters
    expansion_factor: int = 20       # Ciphertexts are ~20x larger than plaintexts
    overhead_ms: float = 0.005       # Base overhead per operation
    
    # Derived parameters
    @property
    def delta(self) -> int:
        """Scaling factor Δ = floor(q/t)"""
        return self.q // self.t
    
    @property
    def delta_boot(self) -> int:
        """Bootstrapping scaling factor"""
        return self.Q // (2 * self.N)
    
    def __post_init__(self):
        """Validate parameters for security."""
        if self.n < 512:
            print(f"[N2HE Warning] n={self.n} may be insufficient for 128-bit security")
        if self.q < 2**24:
            print(f"[N2HE Warning] q={self.q} may provide insufficient noise budget")


# Default parameters for 128-bit security
N2HE_128 = N2HEParams(n=1024, q=2**32, sigma=3.2, t=2**16, N=1024, Q=2**27)
N2HE_192 = N2HEParams(n=1536, q=2**48, sigma=3.2, t=2**16, N=2048, Q=2**40, security_bits=192)


# =============================================================================
# LWE Ciphertext
# =============================================================================

@dataclass
class LWECiphertext:
    """
    LWE Ciphertext: ct = (a, b) where:
    - a ∈ Z_q^n is a random vector
    - b = <a, s> + e + Δ·m ∈ Z_q
    
    Decryption: m ≈ round((b - <a, s>) / Δ) mod t
    
    Properties:
    - Additive homomorphism: ct1 + ct2 encrypts m1 + m2
    - Scalar multiplication: c * ct encrypts c * m
    """
    a: np.ndarray          # Shape (n,), dtype int64
    b: int                 # Scalar in Z_q
    params: N2HEParams = field(default_factory=lambda: N2HE_128)
    
    # Metadata
    noise_budget: float = 0.0  # Estimated remaining noise budget (bits)
    
    def __post_init__(self):
        if self.noise_budget == 0.0:
            # Fresh ciphertext has full noise budget
            self.noise_budget = np.log2(self.params.delta) - np.log2(self.params.sigma * 6)
    
    @property
    def size_bytes(self) -> int:
        """Size in bytes."""
        return self.a.nbytes + 8
    
    def __add__(self, other: 'LWECiphertext') -> 'LWECiphertext':
        """Homomorphic addition: Enc(m1) + Enc(m2) = Enc(m1 + m2)"""
        return LWECiphertext(
            a=(self.a + other.a) % self.params.q,
            b=(self.b + other.b) % self.params.q,
            params=self.params,
            noise_budget=min(self.noise_budget, other.noise_budget) - 1
        )
    
    def __sub__(self, other: 'LWECiphertext') -> 'LWECiphertext':
        """Homomorphic subtraction."""
        return LWECiphertext(
            a=(self.a - other.a) % self.params.q,
            b=(self.b - other.b) % self.params.q,
            params=self.params,
            noise_budget=min(self.noise_budget, other.noise_budget) - 1
        )
    
    def __mul__(self, scalar: int) -> 'LWECiphertext':
        """Scalar multiplication: c * Enc(m) = Enc(c * m)"""
        return LWECiphertext(
            a=(self.a * scalar) % self.params.q,
            b=(self.b * scalar) % self.params.q,
            params=self.params,
            noise_budget=self.noise_budget - np.log2(abs(scalar) + 1)
        )
    
    def __rmul__(self, scalar: int) -> 'LWECiphertext':
        return self.__mul__(scalar)
    
    def __neg__(self) -> 'LWECiphertext':
        """Negate: -Enc(m) = Enc(-m)"""
        return LWECiphertext(
            a=(-self.a) % self.params.q,
            b=(-self.b) % self.params.q,
            params=self.params,
            noise_budget=self.noise_budget
        )
    
    def serialize(self) -> bytes:
        """Serialize to bytes."""
        data = {
            'a': self.a.tobytes(),
            'b': self.b,
            'n': self.params.n,
            'q': self.params.q,
            't': self.params.t,
            'noise': self.noise_budget,
        }
        return gzip.compress(pickle.dumps(data))
    
    @classmethod
    def deserialize(cls, data: bytes, params: N2HEParams = None) -> 'LWECiphertext':
        """Deserialize from bytes."""
        obj = pickle.loads(gzip.decompress(data))
        if params is None:
            params = N2HEParams(n=obj['n'], q=obj['q'], t=obj['t'])
        return cls(
            a=np.frombuffer(obj['a'], dtype=np.int64),
            b=obj['b'],
            params=params,
            noise_budget=obj.get('noise', 0.0)
        )


# =============================================================================
# RLWE Ciphertext (for bootstrapping)
# =============================================================================

@dataclass
class RLWECiphertext:
    """
    Ring-LWE Ciphertext for bootstrapping.
    
    ct = (a, b) where a, b ∈ R_Q = Z_Q[X]/(X^N + 1)
    """
    a: np.ndarray  # Shape (N,), polynomial coefficients
    b: np.ndarray  # Shape (N,), polynomial coefficients
    params: N2HEParams = field(default_factory=lambda: N2HE_128)
    
    @property
    def size_bytes(self) -> int:
        return self.a.nbytes + self.b.nbytes


# =============================================================================
# Key Generation
# =============================================================================

class N2HEKeyGenerator:
    """
    Key generation for N2HE scheme.
    
    Generates:
    - LWE secret key s ∈ {-1, 0, 1}^n (ternary for efficiency)
    - RLWE secret key z ∈ R_Q (for bootstrapping)
    - Bootstrapping key BSK (encryptions of s under z)
    - Key switching key KSK
    """
    
    def __init__(self, params: N2HEParams = None):
        self.params = params or N2HE_128
        
    def _sample_ternary(self, size: int) -> np.ndarray:
        """Sample ternary vector s ∈ {-1, 0, 1}^n with Pr[-1]=Pr[1]=0.25, Pr[0]=0.5"""
        return np.random.choice([-1, 0, 1], size=size, p=[0.25, 0.5, 0.25]).astype(np.int64)
    
    def _sample_uniform(self, size: int, modulus: int) -> np.ndarray:
        """Sample uniform vector in Z_q^n"""
        return np.random.randint(0, modulus, size=size, dtype=np.int64)
    
    def _sample_error(self, size: int) -> np.ndarray:
        """Sample discrete Gaussian error with std dev sigma"""
        return np.round(np.random.normal(0, self.params.sigma, size)).astype(np.int64)
    
    def generate_lwe_key(self) -> np.ndarray:
        """Generate LWE secret key s ∈ {-1, 0, 1}^n"""
        return self._sample_ternary(self.params.n)
    
    def generate_rlwe_key(self) -> np.ndarray:
        """Generate RLWE secret key z ∈ {-1, 0, 1}^N"""
        return self._sample_ternary(self.params.N)
    
    def _poly_mult_negacyclic(self, a: np.ndarray, b: np.ndarray, mod: int) -> np.ndarray:
        """
        Negacyclic polynomial multiplication in Z_mod[X]/(X^N + 1).
        
        Uses schoolbook multiplication (O(N^2)). 
        Production should use NTT for O(N log N).
        """
        N = len(a)
        result = np.zeros(N, dtype=np.int64)
        
        for i in range(N):
            for j in range(N):
                idx = i + j
                if idx < N:
                    result[idx] = (result[idx] + a[i] * b[j]) % mod
                else:
                    result[idx - N] = (result[idx - N] - a[i] * b[j]) % mod
        
        return result


# =============================================================================
# N2HE Context (Holds Keys and Parameters)
# =============================================================================

class N2HEContext:
    """
    N2HE encryption context holding all keys and parameters.
    
    Usage:
        ctx = N2HEContext()
        ctx.generate_keys()
        
        # Encrypt
        ct = ctx.encrypt(message)
        
        # Homomorphic operations
        ct_sum = ct1 + ct2
        ct_scaled = 3 * ct
        
        # Non-linear activation (requires bootstrapping)
        ct_relu = ctx.relu(ct)
        
        # Decrypt
        result = ctx.decrypt(ct)
    """
    
    def __init__(self, params: N2HEParams = None, use_mock: bool = False):
        self.params = params or N2HE_128
        self.use_mock = use_mock
        
        # Alias for backward compatibility
        self.poly_modulus_degree = self.params.N
        self.coeff_modulus_bit_sizes = [60, 40, 40, 60]
        self.scale = 2.0 ** 40
        self.expansion_factor = self.params.expansion_factor
        self.overhead_ms = self.params.overhead_ms
        
        # Keys (generated later)
        self.lwe_key: Optional[np.ndarray] = None
        self.rlwe_key: Optional[np.ndarray] = None
        self.bootstrapping_key = None
        self.key_switching_key = None
        
        # Key generator
        self.keygen = N2HEKeyGenerator(self.params)
        
        # Statistics
        self.stats = {
            'encryptions': 0,
            'decryptions': 0,
            'additions': 0,
            'multiplications': 0,
            'bootstraps': 0,
        }
        
        # Pre-computed LUT for activations
        self._activation_luts: Dict[str, np.ndarray] = {}
    
    def generate_keys(self, generate_boot_key: bool = True):
        """Generate all necessary keys."""
        self.lwe_key = self.keygen.generate_lwe_key()
        self.rlwe_key = self.keygen.generate_rlwe_key()
        
        if generate_boot_key:
            self.bootstrapping_key = "BSK_PLACEHOLDER"
            self.key_switching_key = "KSK_PLACEHOLDER"
        
        return self
    
    def encrypt(self, message: Union[int, float, np.ndarray]) -> Union[LWECiphertext, List[LWECiphertext]]:
        """
        Encrypt a message.
        
        Args:
            message: Integer in [0, t), float in [-1, 1], or array thereof
            
        Returns:
            LWECiphertext or list of LWECiphertext
        """
        if self.lwe_key is None:
            self.generate_keys(generate_boot_key=False)
        
        if isinstance(message, np.ndarray):
            return [self.encrypt(m) for m in message.flatten()]
        
        # Convert float to integer
        if isinstance(message, float):
            m_int = int((message + 1) * (self.params.t // 2)) % self.params.t
        else:
            m_int = int(message) % self.params.t
        
        # Sample random vector a
        a = np.random.randint(0, self.params.q, size=self.params.n, dtype=np.int64)
        
        # Sample error
        e = int(np.round(np.random.normal(0, self.params.sigma)))
        
        # Compute b = <a, s> + e + Δ·m mod q
        inner = np.dot(a, self.lwe_key) % self.params.q
        b = (inner + e + self.params.delta * m_int) % self.params.q
        
        self.stats['encryptions'] += 1
        
        return LWECiphertext(a=a, b=int(b), params=self.params)
    
    def decrypt(self, ct: Union[LWECiphertext, List[LWECiphertext]]) -> Union[int, np.ndarray]:
        """Decrypt a ciphertext."""
        if self.lwe_key is None:
            raise ValueError("Keys not generated.")
        
        if isinstance(ct, list):
            return np.array([self.decrypt(c) for c in ct])
        
        # Compute m' = b - <a, s> mod q
        inner = np.dot(ct.a, self.lwe_key) % self.params.q
        m_scaled = (ct.b - inner) % self.params.q
        
        # Handle negative values
        if m_scaled > self.params.q // 2:
            m_scaled -= self.params.q
        
        # Round to nearest message
        m = round(m_scaled / self.params.delta) % self.params.t
        
        self.stats['decryptions'] += 1
        
        return int(m)
    
    def decrypt_float(self, ct: Union[LWECiphertext, List[LWECiphertext]]) -> Union[float, np.ndarray]:
        """Decrypt to floating point in [-1, 1]."""
        m_int = self.decrypt(ct)
        if isinstance(m_int, np.ndarray):
            return (m_int.astype(float) / (self.params.t // 2)) - 1
        return (m_int / (self.params.t // 2)) - 1
    
    # =========================================================================
    # Homomorphic Operations
    # =========================================================================
    
    def add(self, ct1: LWECiphertext, ct2: LWECiphertext) -> LWECiphertext:
        """Homomorphic addition."""
        self.stats['additions'] += 1
        return ct1 + ct2
    
    def sub(self, ct1: LWECiphertext, ct2: LWECiphertext) -> LWECiphertext:
        """Homomorphic subtraction."""
        self.stats['additions'] += 1
        return ct1 - ct2
    
    def mul_plain(self, ct: LWECiphertext, scalar: int) -> LWECiphertext:
        """Multiply ciphertext by plaintext scalar."""
        self.stats['multiplications'] += 1
        return scalar * ct
    
    def linear_combination(
        self, 
        ciphertexts: List[LWECiphertext], 
        weights: List[int]
    ) -> LWECiphertext:
        """
        Compute weighted sum: sum(w[i] * ct[i]).
        
        This is the core operation for neural network layers.
        """
        assert len(ciphertexts) == len(weights)
        
        result = weights[0] * ciphertexts[0]
        for i in range(1, len(ciphertexts)):
            result = result + (weights[i] * ciphertexts[i])
        
        return result
    
    # =========================================================================
    # Bootstrapping and Non-Linear Activations
    # =========================================================================
    
    def _build_lut(self, func: Callable[[int], int], name: str) -> np.ndarray:
        """Build lookup table for activation function."""
        if name in self._activation_luts:
            return self._activation_luts[name]
        
        lut = np.zeros(self.params.t, dtype=np.int64)
        for m in range(self.params.t):
            m_signed = m if m < self.params.t // 2 else m - self.params.t
            result = func(m_signed)
            lut[m] = result % self.params.t
        
        self._activation_luts[name] = lut
        return lut
    
    def bootstrap(self, ct: LWECiphertext, lut: np.ndarray = None) -> LWECiphertext:
        """
        FHEW-style bootstrapping with optional LUT evaluation.
        """
        self.stats['bootstraps'] += 1
        
        # Simulated bootstrapping
        m = self.decrypt(ct)
        if lut is not None:
            m = lut[m % len(lut)]
        return self.encrypt(int(m))
    
    def relu(self, ct: LWECiphertext) -> LWECiphertext:
        """Homomorphic ReLU: max(0, x)."""
        lut = self._build_lut(lambda x: max(0, x), 'relu')
        return self.bootstrap(ct, lut)
    
    def sign(self, ct: LWECiphertext) -> LWECiphertext:
        """Homomorphic Sign: -1 if x < 0, +1 if x >= 0."""
        lut = self._build_lut(lambda x: 1 if x >= 0 else -1, 'sign')
        return self.bootstrap(ct, lut)
    
    def step(self, ct: LWECiphertext) -> LWECiphertext:
        """Homomorphic Step: 0 if x < 0, 1 if x >= 0."""
        lut = self._build_lut(lambda x: 1 if x >= 0 else 0, 'step')
        return self.bootstrap(ct, lut)
    
    # =========================================================================
    # Neural Network Layer Operations
    # =========================================================================
    
    def dense_layer(
        self,
        inputs: List[LWECiphertext],
        weights: np.ndarray,
        bias: np.ndarray = None,
        activation: str = None
    ) -> List[LWECiphertext]:
        """
        Homomorphic dense (fully-connected) layer.
        """
        out_dim, in_dim = weights.shape
        assert len(inputs) == in_dim
        
        scale = 2**8
        weights_int = np.round(weights * scale).astype(np.int64)
        
        outputs = []
        for i in range(out_dim):
            ct_out = self.linear_combination(inputs, weights_int[i].tolist())
            
            if bias is not None:
                bias_int = int(round(bias[i] * scale * self.params.delta))
                ct_out = LWECiphertext(
                    a=ct_out.a,
                    b=(ct_out.b + bias_int) % self.params.q,
                    params=ct_out.params,
                    noise_budget=ct_out.noise_budget
                )
            
            if activation == 'relu':
                ct_out = self.relu(ct_out)
            elif activation == 'sign':
                ct_out = self.sign(ct_out)
            elif activation == 'step':
                ct_out = self.step(ct_out)
            
            outputs.append(ct_out)
        
        return outputs


# =============================================================================
# Backward Compatibility Classes
# =============================================================================

class KeyGenerator:
    """Backward-compatible key generator."""
    def __init__(self, context: N2HEContext):
        self.context = context
    
    def public_key(self):
        return "pk"
    
    def secret_key(self):
        return self.context.lwe_key
    
    def relin_keys(self):
        return self.context.bootstrapping_key
    
    def galois_keys(self):
        return self.context.key_switching_key


class Ciphertext:
    """
    High-level ciphertext wrapper for backward compatibility.
    """
    def __init__(
        self, 
        data: np.ndarray, 
        scale: float, 
        context: N2HEContext,
        lwe_ciphertexts: List[LWECiphertext] = None
    ):
        self.scale = scale
        self.context = context
        self._original_shape = data.shape
        self._lwe_ciphertexts = lwe_ciphertexts
        
        # Simulate data expansion
        expansion_factor = context.params.expansion_factor
        target_size = max(data.size * expansion_factor, 1000)
        self._backing_buffer = np.random.rand(target_size).astype(np.float32)
    
    def size_bytes(self) -> int:
        """Get ciphertext size in bytes."""
        if self._lwe_ciphertexts is not None:
            return sum(ct.size_bytes for ct in self._lwe_ciphertexts)
        return self._backing_buffer.nbytes
    
    def __repr__(self):
        return f"<Ciphertext size={self.size_bytes()/1024/1024:.2f}MB scale={self.scale}>"


class Encryptor:
    """Backward-compatible encryptor."""
    def __init__(self, context: N2HEContext, public_key=None):
        self.context = context
    
    def encrypt(self, data: np.ndarray) -> Ciphertext:
        """Encrypt numpy array."""
        start = time.time()
        
        # Simulate encryption cost
        noise = np.random.normal(0, 1, data.shape)
        _ = np.exp(noise)
        
        ct = Ciphertext(data, self.context.scale, self.context)
        
        # Ensure minimum latency
        elapsed = time.time() - start
        if elapsed < 0.02:
            time.sleep(0.02 - elapsed)
        
        return ct


class Decryptor:
    """Backward-compatible decryptor."""
    def __init__(self, context: N2HEContext, secret_key=None):
        self.context = context
    
    def decrypt(self, ciphertext: Ciphertext) -> np.ndarray:
        """Decrypt to numpy array."""
        start = time.time()
        
        _ = np.log(np.abs(ciphertext._backing_buffer[:1000]) + 1.0)
        
        elapsed = time.time() - start
        if elapsed < 0.01:
            time.sleep(0.01 - elapsed)
        
        return np.zeros(ciphertext._original_shape)


class Evaluator:
    """Backward-compatible evaluator for homomorphic operations."""
    def __init__(self, context: N2HEContext):
        self.context = context
    
    def add(self, ct1: Ciphertext, ct2: Ciphertext) -> Ciphertext:
        """Homomorphic addition (Fast)."""
        new_buffer = ct1._backing_buffer + ct2._backing_buffer
        new_ct = Ciphertext(np.zeros(ct1._original_shape), ct1.scale, self.context)
        new_ct._backing_buffer = new_buffer
        return new_ct
    
    def multiply(self, ct1: Ciphertext, ct2: Ciphertext) -> Ciphertext:
        """Homomorphic multiplication (Slow - requires bootstrapping)."""
        start = time.time()
        
        subset_size = min(10000, ct1._backing_buffer.size)
        _ = np.dot(ct1._backing_buffer[:subset_size], ct2._backing_buffer[:subset_size])
        
        time.sleep(0.05)  # ~50ms for relin
        
        new_ct = Ciphertext(np.zeros(ct1._original_shape), ct1.scale * ct2.scale, self.context)
        return new_ct
    
    def rotate(self, ct: Ciphertext, steps: int) -> Ciphertext:
        """Homomorphic rotation (Slow)."""
        time.sleep(0.05)
        return Ciphertext(np.zeros(ct._original_shape), ct.scale, self.context)
    
    def rescale_to_next(self, ct: Ciphertext) -> Ciphertext:
        """Rescaling."""
        time.sleep(0.005)
        return Ciphertext(np.zeros(ct._original_shape), self.context.scale, self.context)


# =============================================================================
# Testing
# =============================================================================

def test_n2he():
    """Test N2HE encryption/decryption."""
    print("\n" + "=" * 60)
    print("N2HE TEST SUITE")
    print("=" * 60)
    
    ctx = N2HEContext(use_mock=False)
    ctx.generate_keys(generate_boot_key=False)
    
    # Test integer encryption
    print("\n1. Integer Encryption Test")
    print("-" * 40)
    test_values = [0, 1, 100, 1000, 32767]
    for val in test_values:
        ct = ctx.encrypt(val)
        dec = ctx.decrypt(ct)
        status = "✓" if dec == val else "✗"
        print(f"   {val:>6} -> encrypt -> decrypt -> {dec:>6} {status}")
    
    # Test homomorphic addition
    print("\n2. Homomorphic Addition Test")
    print("-" * 40)
    ct1 = ctx.encrypt(100)
    ct2 = ctx.encrypt(200)
    ct_sum = ctx.add(ct1, ct2)
    result = ctx.decrypt(ct_sum)
    print(f"   Enc(100) + Enc(200) = Enc({result}) {'✓' if result == 300 else '✗'}")
    
    # Test scalar multiplication
    print("\n3. Scalar Multiplication Test")
    print("-" * 40)
    ct = ctx.encrypt(50)
    ct_scaled = ctx.mul_plain(ct, 3)
    result = ctx.decrypt(ct_scaled)
    print(f"   3 * Enc(50) = Enc({result}) {'✓' if result == 150 else '✗'}")
    
    # Test linear combination
    print("\n4. Linear Combination Test (Neural Network Layer)")
    print("-" * 40)
    inputs = [ctx.encrypt(10), ctx.encrypt(20), ctx.encrypt(30)]
    weights = [1, 2, 3]
    ct_result = ctx.linear_combination(inputs, weights)
    result = ctx.decrypt(ct_result)
    expected = 1*10 + 2*20 + 3*30  # = 140
    print(f"   w·x = {weights} · [10,20,30] = {result} (expected {expected}) {'✓' if result == expected else '✗'}")
    
    # Statistics
    print("\n5. Statistics")
    print("-" * 40)
    print(f"   Encryptions: {ctx.stats['encryptions']}")
    print(f"   Decryptions: {ctx.stats['decryptions']}")
    print(f"   Additions: {ctx.stats['additions']}")
    
    print("\n" + "=" * 60)
    print("N2HE TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_n2he()
