# SAM Segmentation API Server

AMD GPU対応のSegment Anything Model (SAM)を使用したセグメンテーションAPIサーバー

## セットアップ

### 1. モデルのダウンロード
```bash
./download_model.sh
```

### 2. Dockerイメージのビルドと起動
```bash
docker-compose up --build
```

## API仕様

### ヘルスチェック
```
GET http://localhost:8000/health
```

### セグメンテーション実行
```
POST http://localhost:8000/generate_mask
```

リクエストボディ:
```json
{
  "image": "base64_encoded_image_string",
  "bboxes": [[x, y, width, height], ...],  // オプション
  "multimask_output": false  // オプション
}
```

レスポンス:
```json
{
  "masks": [
    {
      "mask_base64": "base64_encoded_mask_png",
      "area": 12345,
      "bbox": [x, y, width, height],
      "score": 0.95
    }
  ]
}
```

## 環境変数

AMD GPUのバージョンに応じて`HSA_OVERRIDE_GFX_VERSION`を調整してください:
- gfx1030: RX 6800/6900シリーズ
- gfx1100: RX 7900シリーズ
- gfx90c: MI100
- gfx908: MI50/MI60

docker-compose.ymlまたはDockerfile内で設定可能です。

## 注意事項

- モデルファイルは`models/`ディレクトリに配置されます
- AMD GPUのROCmドライバーがホストシステムにインストールされている必要があります
- SAM ViT-B (中サイズ)モデルを使用しています