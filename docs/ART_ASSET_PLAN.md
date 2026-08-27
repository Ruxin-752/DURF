# DURF Kitchen Lab 美术资源方案

## 本轮落地

当前游戏已换成仓库内的程序化像素渲染：32 px 逻辑格、整数倍最近邻缩放、独立厨房设施、双角色方向/手持物、食材、汤和烹饪进度。它不依赖外部二进制素材，因此桌面与后续 Web 版可以先共享同一套视觉语义，也没有新增素材许可证风险。

## 推荐的外部资源

后续若继续精修，优先只采用可再分发的 CC0 / OFL 资源：

| 用途 | 资源 | 许可证 | 建议用法 |
|---|---|---|---|
| 地面、墙、柜台、装饰 | [Kenney Roguelike/RPG Pack](https://kenney.nl/assets/roguelike-rpg-pack) | CC0 | 做环境装饰层，不覆盖锅和食材状态 |
| 室内家具补充 | [Kenney Roguelike Indoors](https://www.kenney.nl/assets/roguelike-indoors) | CC0 | 灯具、桌椅、货架 |
| 按钮、面板、徽章 | [Kenney Pixel UI Pack](https://kenney.nl/assets/pixel-ui-pack) | CC0 | Web 与桌面 UI 九宫格 |
| 订单/食物图标 | [Free Pixel Food](https://henrysoftware.itch.io/pixel-food) | CC0 | 订单卡和装饰，不替换汤的组合状态 |
| 中文像素短标签 | [Fusion Pixel Font](https://github.com/TakWolf/fusion-pixel-font) | SIL OFL-1.1 | 标题/徽章；长反馈正文继续用清晰无衬线字体 |

Kenney 的许可说明见 [Kenney Support](https://www.kenney.nl/support)。公开发布时仍应增加 `THIRD_PARTY_NOTICES.md`，记录资源名称、版本、下载日期、作者和许可证。

## 集成边界

- 保留现有厨师方向、手持物，以及汤的配料/烹饪阶段语义；通用角色包不能直接整套替换。
- 外部资源大多为 16×16；若采用，应统一到 16×16 逻辑图集并只做整数倍最近邻缩放。
- 不使用原版 Overcooked 的商标、Logo、角色、截图、音频或来源不明的搜索图片；公开名称保持 `DURF Kitchen Lab`。
- 下一轮先换环境装饰层和 UI，再单独制作完整角色动画；这是视觉收益最大、回归风险最低的顺序。
