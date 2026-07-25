# NeuScale MICRO 2026 artifact data

The large inputs for the three-hour functional workflow are distributed in the
[`micro26ae_supplementary_files`](https://github.com/XZman/micro26ae_supplementary_files)
repository rather than tracked in NeuSim. The complete reviewer workflow is in
[`../../../micro26ae.md`](../../../micro26ae.md).

## Published archives

| Archive | SHA-256 | Required layout and destination |
| --- | --- | --- |
| `AzureLLMInferenceTrace_code_3h_sampled.zip` | `4bd1a1b0715d894582370a4238195afc2a518a3fbaa0d5ff60ba23bb135fb427` | Root-level `AzureLLMInferenceTrace_code_3h_sampled.csv`; extract to `artifact_eval/micro26/data`. |
| `request_lookup_cache_deepseekv3_azure_3h_v5p_v6e.zip` | `30f0989dddb13c58c7129f1a92521403495344b3f4cb9eb90867df57b506133e` | Top-level `request_lookup_cache_deepseekv3_azure_3h/`; extract to `artifact_eval/micro26`. |

From the NeuSim repository root:

```bash
git clone https://github.com/XZman/micro26ae_supplementary_files \
  ../micro26ae_supplementary_files
printf '%s\n' \
  '4bd1a1b0715d894582370a4238195afc2a518a3fbaa0d5ff60ba23bb135fb427  AzureLLMInferenceTrace_code_3h_sampled.zip' \
  '30f0989dddb13c58c7129f1a92521403495344b3f4cb9eb90867df57b506133e  request_lookup_cache_deepseekv3_azure_3h_v5p_v6e.zip' | \
  (cd ../micro26ae_supplementary_files && sha256sum -c -)
unzip ../micro26ae_supplementary_files/AzureLLMInferenceTrace_code_3h_sampled.zip \
  -d artifact_eval/micro26/data
unzip ../micro26ae_supplementary_files/request_lookup_cache_deepseekv3_azure_3h_v5p_v6e.zip \
  -d artifact_eval/micro26
```

The supplied CSV is a derived subset of Microsoft Azure’s
[Azure LLM inference trace 2024](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md):
it is the exact first three hours of the sampled Code workload used by the
NeuScale experiments. Request timestamps and token-count values were not
otherwise changed. It contains 3,460 requests, from
`2024-05-10 00:00:00.009930+00:00` through
`2024-05-10 02:59:59.386201+00:00`, with columns `TIMESTAMP`, `ContextTokens`,
and `GeneratedTokens`. [`AzurePublicDataset_LICENSE.txt`](AzurePublicDataset_LICENSE.txt)
and [`AzureLLMInferenceDataset2024.md`](AzureLLMInferenceDataset2024.md) retain
the upstream CC BY 4.0 license, source description, and required DynamoLLM
citation. The trace contains request times and token counts, not request text.

The cache contains 2,942 rank-one FleetSim configuration JSON files plus its
coverage manifest for DeepSeekV3-671B, energy and monetary objectives, prefill
and decode, and v5p/v6e NPUs. The validator checks trace identity, coverage,
model, phases, objectives, and NPU versions.

After extraction, verify the inputs with:

```bash
(cd artifact_eval/micro26/data && sha256sum -c SHA256SUMS)
python -m neusim.run_scripts.prepare_micro26ae_sample_cache --validate-only
```

The cache directory and extracted CSV are runtime inputs ignored by git. As an
alternative to `unzip`, the cache validator can safely extract and atomically
publish the cache ZIP:

```bash
python -m neusim.run_scripts.prepare_micro26ae_sample_cache \
  --archive ../micro26ae_supplementary_files/request_lookup_cache_deepseekv3_azure_3h_v5p_v6e.zip
```

Maintainers can recreate a compact release cache from a generated full cache:

```bash
python -m neusim.run_scripts.package_micro26ae_sample_cache \
  --source /path/to/full_request_lookup_cache \
  --trace artifact_eval/micro26/data/AzureLLMInferenceTrace_code_3h_sampled.csv \
  --output /path/to/request_lookup_cache_deepseekv3_azure_3h_v5p_v6e.zip
```
