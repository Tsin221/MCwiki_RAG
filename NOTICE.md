# 数据与素材来源说明

本文件说明本仓库中非源代码内容的来源与许可。源代码以 MIT 许可发布（见 [LICENSE](LICENSE)），
MIT 许可不覆盖下列内容。

## 知识库数据

`data/original_dataset.json` 抓取自中文 Minecraft Wiki（<https://zh.minecraft.wiki/>），
共 8,200 条记录，每条保留原始页面地址（`source_url` 字段）。

该 Wiki 的内容采用 **Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported
（CC BY-NC-SA 3.0）** 许可，且可能附加其他条款。因此使用或再分发这些内容时需要：

- **署名**：注明内容来自 Minecraft Wiki 及其贡献者；
- **非商业**：不得用于商业目的；
- **相同方式共享**：基于这些内容的演绎作品需以相同许可发布。

`data/processed/` 下的分块与索引由上述数据生成，遵循同样的许可。这些产物不在版本库中，
但使用者自行生成后仍受此约束。

`data/evaluation/` 下的评测题集为本项目自建，其中引用的 Wiki 内容同样遵循 CC BY-NC-SA 3.0。

## 美术素材

`frontend/public/minecraft-sunset.png` 用作页面背景，其具体来源未在本仓库中记录。
Minecraft 相关美术素材的权利归 Mojang Studios 及其许可方所有，再分发前请自行确认授权。

## 商标

Minecraft 是 Mojang Studios 的商标。本项目为非官方项目，与 Mojang Studios、Minecraft Wiki
及其运营方均无关联，也未获得其背书或赞助。
