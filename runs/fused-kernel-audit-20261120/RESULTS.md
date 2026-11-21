# Standalone GB10 review of patches 0195, 0197 and 0198

The reference for decode RMSNorm, RoPE and FP4 is the deployed kit at `abe07c7d`. The reference for prompt RMSNorm, prompt RoPE, the compressor and the candidate pool is the unfused native CUDA code from the November 20 benchmark. The earlier 1,024-token Tetris test isolated the drift to 0197. It saved a reply hash and round counts, without intermediate activations. [Original comparison](../fused-bench-20261120/RESULTS.md)

**The standalone GPU test reproduced the 0197 mismatch and isolated it to the reciprocal RMS calculation.** At width 512, changing just the square-root and division instructions fixed all nine differing bytes across 4,096 random rows. At width 128 it fixed both differing bytes. At each width a differing row was replayed through the full fused decode compressor. The old kernel differed by one byte and the revised kernel matched the reference. The repaired norm also matched on constant-magnitude rows covering all 65,536 bf16 encodings at each width with the model's epsilon.

## Final 0197 decode compressor

Set `TF_DS_FUSED_COMPRESS2_NORM=1` to run 0127’s ring scatter, partner resolution and pair softmax together with the pinned kit RMSNorm in one launch at widths 128 and 512. This takes precedence over the separate 0127 switch. At width 512, the earlier repaired 0197 was slightly slower than 0127 plus the kit at one row. The final kernel ran faster at every measured shape on both runs. [First timing run](decode.log) · [Repeat](decode-repeat.log)

| Decode shape | 0127 + kit µs | First repaired 0197 µs | Final 0197 µs | Speedup over 0127 + kit |
| --- | ---: | ---: | ---: | ---: |
| 1 × 128 | 3.029 | 2.396 | 1.558 | 1.94× |
| 1 × 512 | 3.476 | 3.542 | 1.752 | 1.98× |
| 7 × 512 | 3.989 | 3.618 | 2.002 | 1.99× |
| 32 × 512 | 6.744 | 3.607 | 2.006 | 3.36× |
| 48 × 512 | 8.476 | 3.922 | 2.265 | 3.74× |

Each thread owns one channel. Its scalar loads and stores are coalesced across the warp, including for padded input rows. It reads the current row’s KV and score once and reuses them from registers if either partner selects that row. Each warp resolves both partners in registers and uses a direct comparison for one-row decode. Every channel rounds the pair result to bf16 before normalization. Width 512 exchanges channel values in shared memory to recover the kit’s four-adjacent-channel reduction order. Width 128 skips that exchange. Both widths keep the rounded channel in a register for the weighted output and compute the reciprocal from the four warp sums.

The tuning sweep compared 128, 256 and 512 threads on the same width-512 input. At seven rows the first wide variant took 4.116, 3.029 and 2.710 µs respectively. Retaining rounded channels and reusing current-row loads reduced the final time to 2.002 µs. Timings above are medians from the second of two independent runs with permuted slots and one current and one cached partner per row. Standalone CUDA graphs repeatedly access the same buffers. Serving throughput, the two-rank Tetris hash and the hardware performance ceiling are unmeasured.

Both runs compared all bytes in the output, stride padding, both raw rings and the invalid flag against 0127 plus the pinned kit. Additional cases include two cached partners, two current partners from other blocks, input strides of D+1, negative and out-of-range indices, and extreme logits. Tests crossed both the 32- and 64-row lookup boundaries. The original full probe also passed both mismatch witnesses, the unfused compressor comparison and its 0195/0198 checks with the final sources. [Decode probe](decode.py) · [Full regression log](full.log)

The final recipe and benchmark port were reapplied in clean checkouts and matched the reviewed source trees. All nine focused Zig host tests passed. Both ARM64 programs used by the benchmark image, `tensorfold-native` and `tf-dsv41-lanes`, linked and embedded the exact SM121 compressor and ops fatbins used for the integration build. The unrelated test-module fatbins used for linking are described under build coverage below.

### Initial repair and isolated GPU performance

Median microseconds per operation, measured on the head after the other campaign restored production. Each sample was a graph of 32 consecutive operations. Nine samples per arm rotated launch order. Both independent runs passed the output checks. The table uses the second run. Repeated inputs can be cache-resident. These measurements establish the cost of the exercised kernels. The revised two-rank model's Tetris hash and serving throughput remain unmeasured. [First run](gb10.log) · [Repeat and extended parity checks](gb10-repeat.log)

| Operation | Shape | Unfused µs | Old fused µs | Revised µs |
| --- | --- | ---: | ---: | ---: |
| 0197 decode compressor | 7 × 512 | 11.106 | 4.751 | 3.669 |
| 0197 decode compressor | 32 × 512 | 12.050 | 7.524 | 3.665 |
| 0197 prompt compressor | 128 × 512 | 6.417 | 7.783 | 2.380 |
| 0197 prompt compressor | 1,024 × 512 | 13.652 | 15.247 | 7.820 |
| 0195 indexer list, random scores | 7 rows, 4,096 blocks, k = 2,048 | 44.766 | 45.584 | 9.804 |
| 0195 indexer mask, many ties | 128 rows, 4,096 blocks, k = 2,048 | 158.767 | 154.486 | 39.190 |
| 0198 decode indexer key | 7 rows | 2.973 | 3.399 | 2.139 |
| 0198 decode latent | 7 rows | 2.132 | 3.035 | 1.617 |
| 0198 decode query | 7 rows × 32 heads | 2.199 | 9.303 | 1.681 |
| 0198 prompt indexer key | 128 rows | 4.311 | 3.543 | 3.024 |
| 0198 prompt latent | 128 rows | 3.606 | 4.123 | 1.868 |

0127 plus the kit norm is a stronger decode compressor control than the fully unfused production sequence. It took 4.127 µs at seven rows and 6.948 µs at 32. At one row it took 3.400 µs against the revised 3.475 µs. The short 0195 test with k equal to the block count exercises a boundary the served candidate-source dispatcher skips.

## 0197 arithmetic and dispatch

| Code reviewed | Finding and change |
| --- | --- |
| Decode row addresses and ring updates | Each block scatters its input to its assigned raw slot. The first repair searched one gather index per warp. The final version searches both in each warp, using the first matching current row before reading previous ring contents. Source and destination strides are in elements. This matches 0127 for the distinct raw slots the scheduler supplies. Multiple bad gathers can mark the same invalid flag, so the fused path now sets it atomically. |
| Pair softmax and bf16 staging | Both paths use the same max, `expf`, RN divisions, products, four-term sum and bf16 rounding as `tf_ds_compress2_kernel`. The reduction consumes those rounded values. Extra prompt threads compute the independent channels while the x dimension retains the original reduction lanes. Sweeping 128, 256 and 512 threads at width 512 favored 512 for 1–32 rows, 256 at 128 rows and 128 at 1,024 rows. |
| Decode sum of squares | In the 512-wide kit cubin `ee9e02…`, instructions 0x1c0–0x200 square element 1 first and accumulate elements 0, 2 and 3 with FFMA. The patch previously used sequential separately rounded multiplies and adds. For normal bf16 inputs the individual square is exactly representable in fp32. The revised code follows the instruction order, including its rounding for extreme inputs. The warp XOR offsets 16, 8, 4, 2, 1 and inter-warp offsets 2, 1 already matched. |
| Mean, epsilon and reciprocal | Instructions 0x460–0x4f0 scale the sum and add epsilon with FFMA, then use MUFU.SQRT and MUFU.RCP with range handling. The old implementation used `sqrtf` and `__fdiv_rn`. The revised helper spells out the approximate PTX instructions. It retains the separate fp32 products by the row value and weight and the final bf16 conversion. The 128-wide specialization `0cca97…` uses the same tail. Other decode widths return to the original dispatcher. |
| Prompt norm | The four accumulators, shared fold, shuffle-down, row-dependent lane count, host mean factor and rsqrt match the existing torch path. At width 2048 a single row needs 512 lanes. The original standalone 0197 provided 128 scratch floats and relied on 0198 to enlarge them. The repair puts 512 floats in 0197 itself, including the host reference. |
| Zig integration | Checked switch parsing, symbol lookup, argument order/types, launch dimensions, PDL, stream propagation, ratio-1 bypass, zero-row handling and 0127 precedence against `Exact`, `compressorChain` and `kvSourceUpdate`. The optional compressor stream still owns every access in its chain. |

## 0195 selection and pool writes

The candidate-source calls receive the compressed score width. With the checkpoint's 2,048 candidate blocks of eight scores, the expensive selection engages only above 16,384 score columns. The short 1,024-token reply benchmark provides no timing evidence about that regime.

| Code reviewed | Finding and change |
| --- | --- |
| Block maxima | Keep the first maximal score, canonical NaN, short final block and newest-block +infinity override. A nonpositive visible count pins no block. Save the maxima for subsequent reads in CTA shared memory and retain the original scratch output. |
| Selection and ties | The original kernel appended winning columns with atomics and sorted up to 2,048 of them in shared memory. A warp searches the radix histogram using eight bins per lane. It skips two constant column bytes and stops when an entire bin fills the remaining quota. The inclusive cutoff on the unique integer score/column keys determines membership. Warp ballots, population counts and an ordered prefix sum assign each winner its ascending output index directly. |
| Mask and list | Each column has one mask writer. The selected-column list and the optional block list keep their original order, visibility filter and NaN/-infinity handling. Preserve the prompt mask offset, row stride and unselected bytes outside the active width. |
| Launcher and fallback | Validate the block size before division, use overflow-safe ceiling division, check strides, and request four bytes of shared memory per maximum. The largest supported row takes 32 KiB of dynamic shared memory. Wider rows retain the existing path. The long recipe also needs a forward declaration for the existing greedy argmax helper to compile this CUDA translation unit. |

## 0198 rotation, normalization and packing

| Code reviewed | Finding and change |
| --- | --- |
| Decode and prompt keys | The decode key now shares the repaired 128-wide kit norm. The prompt key retains torch's reduction and lane count. Both preserve the bf16 boundary before rotating, the rotated key buffer, target row indirection and separate code and scale outputs. |
| RoPE coefficients | Checked cos/sin addressing, rotary suffix and pair order. Decode contracts the even value times sine into the imaginary part. Prompt contracts the odd value times cosine there. Preserve both forms. |
| Scale and E2M1 conversion | The prior code gave one thread each 16- or 32-element scale group. Lanes within a half warp or warp now reduce the absolute maximum cooperatively. The kit uses numeric min/max for NaN operands. The old NaN-propagating CUDA helpers disagreed. For E4M3 latents, hardware rounds the scale to FP8 and lanes compare bf16 magnitudes with seven exactly representable scaled midpoints. All 8,257,536 input/positive-scale pairs matched the independent RN-division reference. The GB10 cache bytes also matched the actual kit with every bf16 encoding represented. Key and query paths keep RN division, scale floors, exponent rounding, packing order and signed-zero behavior. |
| Decode query | Each lane reads one element and exchanges its partner by shuffle. Compute the rotated bf16 value once, then quantize in registers with a warp maximum. Preserve both the in-place rotated input and dequantized bf16 output. |
| Prompt latent | Copy directly from the original latent into the fused kernel, rotate into the original destination and write the packed cache. This removes a separate copy launch and retains both buffers' contents. Both latent paths rotate adjacent lane values in registers and pack low/high half-row elements together. The owner check still precedes the prompt latent store. |
| Launchers | Checked the five argument lists against their CUDA signatures, the head/row grids, target and position arrays, strides, scale block sizes and fallback paths. The decode latent still respects the sharded target or trash row. |


## Reproduction and build coverage

Run the probe on a GB10 with the CUDA toolkit, Python 3 and NumPy. Supply the original November 20 engine CUDA source directory (commit `e3eb746`) and a second CUDA source directory with the revised kernels. On the long recipe tree, apply patches 0195, 0197 and 0198 in order after the reworked 0193/0194. For the original benchmark tree at `e3eb746`, the incremental port applies directly. [Production-source benchmark port](production-port.patch)

```bash
bash build.sh "$old_engine/zig/kernels/cuda" "$new_engine/zig/kernels/cuda" "$scratch/binaries"
OPENBLAS_NUM_THREADS=1 python3 probe.py "$kit/aot/cubins" "$scratch/binaries"
python3 decode.py "$kit/aot/cubins" "$scratch/binaries/new-torch.cubin"
```

The decode probe accepts `--previous <first-repair.cubin>` to include the intermediate 128-thread 0197 as a third timing arm. The probe uses the original independent kernels for outputs and CUDA events for graph timings. It checks the compressed rows, both rings, flags, stride padding, maxima, selected columns, mask, block list, rotated and dequantized bf16 values, packed codes, and scales. Cases include partial blocks, the maximum fused width, a non-power-of-two k, ties, NaNs, infinities, signed zero, permuted cache destinations, and a mix of cached and current ring partners. The full-chain comparison uses valid slot indices as generated by the scheduler. The isolated 0127 comparison also covered an invalid partner during tuning. [Build script](build.sh) · [Probe](probe.py)

The exported patches apply cleanly to the recipe base and reproduce the reviewed tree byte for byte. A build of the full long-series server encounters pre-existing errors in `reply_text.zig` and `http_body.zig`. The production-source benchmark port cross-built both ARM64 executables successfully. Byte searches confirmed that each executable embeds both newly compiled SM121 fatbins. This was a link check using cached fatbins for the other modules, including placeholders for the unrelated probe/GDN test modules. The local host suite reported 615 of 617 executed tests passing, with additional suites cached. The two failures were the previously recorded `prompt_cache.test.0178` and `round.hotPick`. The service ran its production image throughout the standalone checks.
