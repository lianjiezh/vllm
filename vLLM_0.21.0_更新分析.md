# vLLM 0.21.0 版本详细分析报告

发布日期：2026年5月15日  
提交数：367个 commits  
贡献者：202位（含49位新贡献者）

---

## 一、核心亮点

### 1.1 破坏性变更（Breaking Changes）

#### 1.1.1 Transformers v4 正式弃用
- **PR**：#40389
- **描述**：vLLM 0.21.0 正式弃用 `transformers` v4 支持，要求用户迁移到 v5
- **影响范围**：
  - 所有仍依赖 Transformers v4 的项目需要评估兼容性
  - 老插件、老模型封装、魔改 tokenizer 可能受到影响
  - 建议在升级前进行全面测试

#### 1.1.2 C++20 构建要求
- **PR**：#40380
- **描述**：vLLM 现在需要 C++20 兼容编译器，以与 PyTorch 保持兼容
- **影响范围**：
  - **构建环境**：源码编译、内网离线构建、魔改 vLLM 的用户受影响最大
  - **预编译 wheel**：对直接使用 `pip install` 的用户影响较小
  - **企业环境**：内网机器编译器版本可能需要升级

### 1.2 重要功能更新

#### 1.2.1 KV Offload + 混合内存分配器（HMA）深度整合
- **PR**：#41228, #41445, #39571, #40900, #41549
- **关键改进**：
  - 调度器端滑动窗口组支持
  - 完整 HMA 启用
  - 多连接器 HMA 支持
  - 每作业存储完成追踪
  - OffloadingConnector 中的 DCP/PCP 支持
  - MooncakeStoreConnector 用于分布式 KV 卸载
- **实际价值**：显著提升长上下文、多并发、推理模型场景下的显存管理效率

#### 1.2.2 推测解码支持思考预算（Thinking Budget）
- **PR**：#34668
- **描述**：推测解码现在能正确处理推理/思考预算，使推理模型的 spec decode 工作正常
- **意义**：
  - 解决了推理模型时代的关键兼容性问题
  - 支持 DeepSeek-R1 等具有推理特性的模型正确使用推测解码
  - 使 vLLM 真正适配"会思考"的模型

#### 1.2.3 Blackwell 的 TOKENSPEED_MLA 后端
- **PR**：#41778
- **适用模型**：DeepSeek-R1、Kimi-K25
- **场景**：Prefill + Decode 阶段
- **价值**：在 Blackwell GPU 上提供专用优化的注意力后端

---

## 二、模型支持更新

### 2.1 新增架构
| 模型 | PR | 说明 |
|------|-----|------|
| MiMo-V2.5 | #40967, #41905 | 多模态模型，支持 MTP 推测解码 |
| Laguna XS.2 | #41129, #41880 | 新架构支持 |
| Moondream3 | #32325 | 视觉语言模型 |
| Qianfan-OCR | #40136 | OCR 模型 |
| Cohere MoE | #40817 | Cohere MoE 模型 |
| Cohere Eagle | #42078 | Cohere 推测解码模型 |

### 2.2 推测解码增强
- **EAGLE for Mistral** (#41024)
- **Gemma4 MTP** (#41745)
- **MiMo-V2.5 MTP** (#41905)
- **Cohere Eagle** (#42078)

### 2.3 DeepSeek V4 专项优化
- AMD/ROCm 支持 (#40871)
- 流水线并行 (#41694)
- `max` 推理强度 (#40982)
- 分解服务修复 (#41957)

### 2.4 工具调用增强
- Cohere 推理和工具解析器 (#40422)
- LFM2/2.5 工具解析器 (#39243)

### 2.5 Gemma3/Gemma4 改进
- `hidden_act` 变体支持 (#40588)
- 流水线并行修复 (#40786)
- MoE 修复 (#41206, #41574, #41401)
- 工具解析器崩溃修复 (#41991, #42188)

### 2.6 其他模型增强
- Model Runner V2：Qwen3.5/Mamba 混合模型支持 (#35520)，`logprob_token_ids` 支持 (#40559)
- CUDA graph：Qwen2.5-VL 的 ViT CUDA graph 支持 (#40830)
- 兼容性：Transformers v5 的 Vendor HCXVisionConfig (#38447)，传统 `rope_type` checkpoint 支持 (#41734)

---

## 三、引擎核心改进

### 3.1 KV Offloading + HMA 深度整合
- 调度器端滑动窗口组 (#41228)
- 完整 HMA 启用 (#41445)
- 多连接器 HMA (#39571)
- 每作业存储完成 (#39186)
- OffloadingConnector 中的 DCP/PCP 支持 (#41549)
- MooncakeStoreConnector 用于分布式 KV 卸载 (#40900)

### 3.2 推测解码优化
- 思考预算支持 (#34668)
- 独立的草稿模型注意力后端选择 (#39930)
- 多模态模型支持（带警告）(#41752)
- 消除每步分配 (#41043)

### 3.3 Model Runner V2 改进
- 拒绝采样接受率修复 (#40651)
- Draft prefill 前跳过元数据重建 (#40410)
- Draft decode 步骤间重建元数据 (#41162)
- Qwen3.5/Mamba 混合支持 (#35520)

### 3.4 其他核心改进
- **路由**：用设备缓存和异步 D2H 流水线替换路由回放 (#39917)
- **Ray**：默认启用 RayExecutorV2 (#41421)，DP>1 时的 actor 名称冲突修复 (#40398)
- **稳定性**：两阶段暂停防止调度器死锁 (#39366)，线程安全的 HF tokenizer 包装 (#41181)，模型加载期间通过 max_split_size_mb 防止 OOM (#41268)
- **IndexCache 支持**：DSA 模型的 IndexCache 支持 (#37735)

---

## 四、硬件与性能优化

### 4.1 NVIDIA Blackwell
- TOKENSPEED_MLA 后端用于 DSR1/Kimi-K25 (#41778)
- 更快的每 token FP8 组量化打包 kernel (#41326)
- NVIDIA Thor/SM110 上的 FP8 (#39712)
- 非兼容大小的 CUTLASS 缩放矩阵乘法 (#41868)

### 4.2 通用性能优化
- FlashInfer top-k/top-p 采样器默认启用 (#40376)
- ViT 的 FP8 FlashInfer 注意力 (#38065)
- TurboQuant 共享反量化缓冲区 (#40941)
- AllPool.forward 提速 51% (#41163)
- 消除 pooling (#41433) 和 attention (#41434) 中的 GPU<->CPU 同步
- numpy 零拷贝嵌入序列化 (#41681)
- 纯文本时跳过多模态处理器 (#41246)
- FlashInfer FP8 异步 TP 融合 (#39505)
- NVFP4 全收集 GEMM 融合用于 AsyncTP (#41882)
- 重新启用 DP/PP 的 allreduce+RMS 融合 (#41458)
- DeepSeek bf16→fp32 通过 torch.mm (#41300)
- 稀疏后端的持久化 MLA (#41990)
- 可配置的 safetensors checkpoint 预取 (#41499)
- 融合 mhc_post_pre kernel (#41536)
- 2D-grid W8W8 组量化 kernel (#42153)
- KV cache 交换的宽松内存排序 (#39306)

### 4.3 AMD ROCm
- ROCm 7.2.2 (#41386)
- DBO（动态批处理优化）(#34726)
- AITER 融合 Allreduce+RMSNorm (#37646)
- Qwen3-Next 的融合共享专家（FSE）(#39280)
- DeepSeek V3.2 TP4 AITER MLA (#41835)
- GDN 线性注意力融合 (#40711)
- 消除 AITER 中冗余的 MoE 缓冲区复制 (#41713)
- CPU offloading 支持 (#40549)
- DeepEP API 更新 (#39721)
- 限制 Triton paged attention 块大小以修复共享内存 OOM (#38502)

### 4.4 CPU 优化
- AMX/AVX-512 的 FP8 注意力 (#39445)
- FP8 W8A16 线性层 (#41186)
- FP8 W8A16 MoE (#41314)
- DNNL AVX2 W8A8 Int8 (#41318)
- Qwen 3.5/3.6 的门控 DeltaNet 注意力 (#41025)
- RISC-V OMP 线程自动绑定 (#40569)

### 4.5 其他硬件支持
- **Intel XPU**：Top-k/top-p 采样 kernel (#39285)，out-of-place all-reduce (#41808)，LoRA 支持 (#38206)
- **IBM Power**：VSX attention backend (#40451)
- **FlexAttention**：为 batch invariant 模式重新启用 (#40842)
- **MLA**：抽象 MLA prefill 后端，消除 cuDNN 依赖 (#32623)

---

## 五、大规模服务支持

### 5.1 分解服务（Disaggregated Serving）
- P 和 D 之间的双向 KV cache 传输 (#32553)
- NIXL 传输重新设计 (#40731)
- EPLB 内存开销优化 (#40013)
- NIXL connector 升级到 1.x (#42364)
- Mooncake KVConnectorStats 用于传输可观测性 (#40414)
- NIXL P-node 预准入拒绝通知 (#41269)
- 跳过的 P-rank 的 KV 块释放 (#40449)

### 5.2 DCP 优化
- 在 DCP A2A 中打包输出和 LSE (#41160)

### 5.3 MoE 改进
- 用于 out-of-tree MoE runners 的 PluggableLayer 接口 (#35178)

### 5.4 LoRA 增强
- 初始专家并行（EP）支持 (#40867)
- Qwen3.5 LoRA 融合修复 (#37912)

---

## 六、量化支持

### 6.1 NVFP4
- KV cache 支持 (#40177)
- Hopper 和 AMD 的 Triton dequant/QDQ 模拟 kernels (#40033)
- Gemma4 的 TRT-LLM NvFP4 融合 MoE 上的 GELU (#41050)
- ModelOpt NVFP4 W4A16 (#41769)
- NVFP4 all-gather GEMM 融合用于 AsyncTP (#41882)
- GLM4-MoE NVFP4 加载修复 (#41755)

### 6.2 MXFP4
- Humming MXFP4 MoE 后端 (#41083)
- FlashInfer CUTLASS MXFP4-MXFP8 MoE 修复 (#42089)

### 6.3 其他量化改进
- **TurboQuant**：混合模型和均匀量化支持 (#39931)
- **Compressed tensors**：允许非显式忽略的配置 (#41965)
- **FP8**：偏置加载修复 (#41424)，为正确性暂时禁用 FlashInfer autotune (#41524)
- **DSV4**：改进的融合 Indexer Q 量化 kernel (#41428)

---

## 七、API 与前端更新

### 7.1 Responses API
- 流式工具/函数调用带 required (#40700)
- 命名工具/函数选择 (#41110)
- 重新提交缺少字段的输出项 (#41355)

### 7.2 OpenAI 兼容性
- 响应中的 `system_fingerprint` 字段 (#40537)
- `prompt_embeds` 内容部分支持 (#40720)
- `defer_loading` 和 `tool_reference` 支持 (#40190)
- 聊天完成响应中的渲染提示文本 (#42052)
- 强制工具选择中容忍空内容 (#40148)

### 7.3 工具调用
- XGrammar 0.2.0 带结构标签用于严格工具调用 + 推理 (#40894)
- Cohere 推理/工具解析器 (#40422)
- LFM2/2.5 工具解析器 (#39243)

### 7.4 其他 API 更新
- **Tokenizer**：Fastokens 支持 (#41741)
- **RLHF**：显式 `/start_weight_update` 和 `/finish_weight_update` APIs (#39212)
- **ASR**：取消时的引擎请求中止 (#41266)
- **配置**：`VLLM_SKIP_MODEL_NAME_VALIDATION` 环境变量 (#34676)，可配置的模型权重加载追踪 (#41086)，Triton JIT 编译监视器 (#40137)

---

## 八、构建与依赖管理

### 8.1 关键变更
- **破坏性**：C++20 要求用于 PyTorch 兼容性 (#40380)
- **破坏性**：Transformers v4 弃用 (#40389)

### 8.2 优化
- 通过延迟 FlashInfer cubin 下载将 Docker 镜像大小减少约 2.5 GB (#41134)
- CUDA 13.0 wheels 切换到 PyTorch manylinux_2_28 基础 (#41416)
- 每个 Python 构建 DeepGEMM 捆绑 wheel 用于 CPython 兼容性 (#41516)
- 嵌入容器镜像来源元数据 (#40653)
- tpu-inference 升级到 v0.19.0 (#41844)
- NIXL connector 升级到 1.x (#42364)
- ROCm 7.2.2 (#41386)

---

## 九、升级建议

### 9.1 立即行动项
1. **评估 Transformers v5 迁移**：检查现有代码、插件、模型封装与 Transformers v5 的兼容性
2. **验证构建环境**：如果进行源码编译，确保编译器支持 C++20
3. **测试 KV offload 配置**：如果使用 KV offload，测试新的 HMA 集成

### 9.2 优先测试的场景
- 推理模型（DeepSeek-R1、Kimi 等）的推测解码
- Blackwell GPU 上的 TOKENSPEED_MLA 后端性能
- 大规模服务的分解服务和 DCP 改进
- 新量化选项（NVFP4、MXFP4）的性能与精度权衡

### 9.3 风险评估
| 变更项 | 风险等级 | 说明 |
|--------|---------|------|
| Transformers v5 迁移 | 中 | 需要充分测试，可能影响自定义代码 |
| C++20 构建要求 | 低（预编译 wheel）/高（源码编译） | 取决于您的部署方式 |
| KV Offload + HMA | 中 | 性能改进但需要验证配置 |
| 推测解码思考预算 | 低 | 主要是功能增强，向后兼容 |

---

## 十、总结

vLLM 0.21.0 是一次**工程化导向的重要版本升级**，核心围绕四个方向：

1. **清理技术债务**：弃用 Transformers v4、升级到 C++20，为未来发展松绑
2. **强化大规模服务能力**：KV offload + HMA 深度整合、分解服务优化
3. **适配推理模型时代**：推测解码支持思考预算，真正服务于"会思考"的模型
4. **紧跟硬件演进**：Blackwell 专用优化、多硬件平台持续支持

对于**生产环境部署**，特别是运行 DeepSeek、Kimi、Qwen 等推理模型或面临显存压力的场景，这是一个**值得升级的版本**。对于**本地单卡偶尔跑模型**的用户，可以先观望。

