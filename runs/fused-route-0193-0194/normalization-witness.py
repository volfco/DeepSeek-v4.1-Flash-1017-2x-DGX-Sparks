"""Check the FP32 incompatibility between the kit's RCP/MUL and patch 0193's DIV.RN."""
import struct


def bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def value(word):
    return struct.unpack("<f", struct.pack("<I", word))[0]


def f32(x):
    return value(bits(x))


# Positive integral square roots avoid softplus in the source's lg > 20 branch.
scores = [14, 9, 8, 7, 6, 5]
denominator = f32(sum(scores) + 1e-20)
assert denominator == 49
numerator = scores[-1]
direct = bits(f32(numerator / denominator) * 1.5)
lower = bits(1.0 / denominator)
upper = lower + 1


def multiply_by_reciprocal(word):
    return bits(f32(numerator * value(word)) * 1.5)


assert (lower, upper) == (0x3CA72F05, 0x3CA72F06)
lo, hi = map(multiply_by_reciprocal, (lower, upper))
assert (lo, direct, hi) == (0x3E1CBC14, 0x3E1CBC15, 0x3E1CBC16)
# Both multiplies and their roundings are monotone for positive inputs.
# No FP32 reciprocal between these adjacent words exists.
print(f"Original logits: {[score**2 for score in scores]}, sum of scores: {denominator:g}")
print(f"0193 direct division:                    0x{direct:08x}")
print(f"Kit multiply by reciprocal 0x{lower:08x}: 0x{lo:08x}")
print(f"Kit multiply by reciprocal 0x{upper:08x}: 0x{hi:08x}")

# A second case retains all six slots under the production top-p rule.
# Five scores tie at 9, so every tie order has the same weight prefix.
scores = [11, 9, 9, 9, 9, 9]
denominator = f32(sum(scores) + 1e-20)
numerator = 9
lower = 0x3C924924
upper = lower + 1
direct = bits(f32(numerator / denominator) * 1.5)
lo, hi = map(multiply_by_reciprocal, (lower, upper))
assert (lo, direct, hi) == (0x3E76DB6C, 0x3E76DB6E, 0x3E76DB6F)
assert sum(scores[:-1]) / denominator < 0.85
print(f"Unpruned logits: {[score**2 for score in scores]}, last exclusive prefix: 47/56")
print(f"0193 direct division:                    0x{direct:08x}")
print(f"Kit multiply by reciprocal 0x{lower:08x}: 0x{lo:08x}")
print(f"Kit multiply by reciprocal 0x{upper:08x}: 0x{hi:08x}")
