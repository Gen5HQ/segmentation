const fs = require('fs');
const path = require('path');

// Modal のエンドポイントURL
const MODAL_ENDPOINT_URL = 'https://itta611--gen5-segmentation-generate-mask.modal.run';

async function segmentImage(imagePath) {
    try {
        // 画像ファイルを読み込み、base64エンコード
        const imageBuffer = fs.readFileSync(imagePath);
        const base64Image = imageBuffer.toString('base64');

        // POSTリクエストの準備
        const response = await fetch(MODAL_ENDPOINT_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: base64Image
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        
        // 結果の処理
        console.log(`見つかったマスク数: ${result.total_masks_found}`);
        
        // マスク画像をファイルに保存
        if (result.masks && result.masks.length > 0) {
            result.masks.forEach((mask, index) => {
                const maskBuffer = Buffer.from(mask.mask_base64, 'base64');
                const maskPath = path.join(path.dirname(imagePath), `mask_${index + 1}.png`);
                fs.writeFileSync(maskPath, maskBuffer);
                console.log(`マスク ${index + 1} を保存: ${maskPath}`);
                console.log(`  - 面積: ${mask.area}`);
                console.log(`  - 安定性スコア: ${mask.stability_score}`);
                console.log(`  - バウンディングボックス: [${mask.bbox.join(', ')}]`);
            });
        }
        
        return result;
        
    } catch (error) {
        console.error('エラーが発生しました:', error);
        throw error;
    }
}

// 使用例
async function main() {
    const imagePath = './sample.jpg';
    
    console.log(`画像をセグメント化中: ${imagePath}`);
    const result = await segmentImage(imagePath);
    console.log('処理完了');
}

// Node.js 18+ の場合（fetch が組み込まれている）
if (require.main === module) {
    main().catch(console.error);
}

// Node.js 17以前の場合は node-fetch をインストール
// npm install node-fetch
// const fetch = require('node-fetch');

module.exports = { segmentImage };