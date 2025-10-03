const fs = require("fs");

async function testEndpoint() {
  try {
    // sample.jpgを読み込んでbase64エンコード
    const imageBuffer = fs.readFileSync("./sample.JPG");
    const base64Image = imageBuffer.toString("base64");

    console.log("画像サイズ:", imageBuffer.length, "bytes");
    console.log("Base64長:", base64Image.length, "characters");

    // エンドポイントに送信
    const response = await fetch(
      "https://itta611--gen5-segmentation-generate-mask.modal.run",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          image: base64Image,
        }),
      }
    );

    console.log("レスポンスステータス:", response.status);
    console.log("レスポンスヘッダー:", response.headers);

    const text = await response.text();
    console.log("レスポンス:", text);

    if (response.ok) {
      const result = JSON.parse(text);
      console.log("成功! マスク数:", result.total_masks_found);
    }
  } catch (error) {
    console.error("エラー:", error);
  }
}

testEndpoint();
