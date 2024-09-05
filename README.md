# Qwen2-VL 多模态处理应用

基于 Qwen2-VL 模型的多模态处理应用，支持图片和视频分析，并提供自定义指令功能。

## 功能特点

- 支持图片和视频输入
- 使用 Qwen2-VL-7B-Instruct 模型进行分析
- 自定义指令支持，允许灵活的查询
- 支持 CUDA、MPS 和 CPU 设备
- 用户友好的 Gradio 界面

## 安装说明

1. 克隆仓库：

   ```
   git clone [您的仓库URL]
   cd [仓库名称]
   ```

2. 创建并激活虚拟环境（推荐）：

   ```
   python -m venv venv
   source venv/bin/activate  # 在 Windows 上使用 venv\Scripts\activate
   ```

3. 安装依赖项：

   ```
   pip install -r requirements.txt
   ```

   注意：确保已安装 PyAV 和 torchvision。如果没有，可以使用以下命令安装：

   ```
   pip install av torchvision
   ```

4. 下载 Qwen2-VL-7B-Instruct 模型：

   请从官方途径下载模型并将其放置在项目根目录下的 `./Qwen2-VL-7B-Instruct` 文件夹中。

## 使用方法

1. 运行应用：

   ```
   python app.py
   ```

2. 在浏览器中打开显示的本地 URL（通常是 `http://127.0.0.1:7860`）。

3. 使用界面：
   - 在 "上传图片" 区域上传一张或多张图片（如需要）。
   - 使用 "上传视频" 组件上传视频文件（如需要）。
   - 在 "输入指令" 文本框中输入您的查询或分析要求。
   - 点击 "提交" 按钮获取模型输出。

## 依赖项

主要依赖项包括：

- gradio
- torch
- transformers
- Pillow
- av (PyAV)
- torchvision

完整的依赖列表请参见 `requirements.txt` 文件。

## 注意事项

- 确保您有足够的磁盘空间和内存来运行大型语言模型。
- 处理大型视频文件可能需要较长时间，请耐心等待。
- 如果遇到 CUDA 相关错误，请确保您的 CUDA 驱动程序与 PyTorch 版本兼容。

## 故障排除

如果遇到 "PyAV is not installed" 或类似的错误，请确保已正确安装所有依赖项。可以尝试重新运行：

```
pip install av torchvision
```

## 贡献

欢迎提交问题报告和拉取请求。对于重大更改，请先开 issue 讨论您想要改变的内容。

## 许可证

本项目采用 MIT 许可证。查看 [LICENSE](LICENSE) 文件了解更多信息。