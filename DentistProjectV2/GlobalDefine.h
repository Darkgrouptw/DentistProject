#pragma once
#define TEST_NO_OCT						// 測試的時候會用到
//#define SHOW_TRCUDAV2_TOTAL_TIME		// 顯示整體時間
//#define SHOW_TRCUDAV2_DETAIL_TIME		// 顯示每個 Detail 的時間
//#define SHOW_TRCUDAV2_TRANSFORM_TIME	// 顯示 Transform 成圖的時間等
#define DISABLE_SINGLE_RESULT			// 關閉顯示真正掃描時單片的結果 (但 Debug 的時候，還是會顯示 Message)
//#define DISABLE_MULTI_RESULT			// 關閉顯示真正掃描時多片的結果 (但 Debug 的時候，還是會顯示 Message)

/*
ToDo
1. 判斷圖片的亮度是否大於某一個 th
(x) 2. 邊界要做 Smooth 的動作
3. zratio 要修改
4. Alignment 的部分要做成動畫，然後慢慢 align 起來
(x) 5. 九軸的資訊要算到點雲內
6. 測試的部分， shake multi 還沒出來

2019/02/12
(x) 7. 測試藍芽的 Establish
8. 在拚時候，要停止 GL 更新
9. 一開始掃描，可能會有黑色的部分
*/