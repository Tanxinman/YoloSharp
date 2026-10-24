using System;
using System.Collections.Generic;
using Data;
using OpenCvSharp;
using YoloSharp.Models;
using YoloSharp.Types;

namespace YoloSharpNet48Demo
{
    /// <summary>
    /// YoloSharp net48 兼容性验证：
    /// 参照 YoloSharpDemo（net6.0），验证 .NET Framework 4.8 下
    /// TorchSharp + libtorch + OpenCvSharp + YoloSharp(netstandard2.0) 推理链路。
    /// </summary>
    internal class Program
    {
        static void Main(string[] args)
        {
            // v9验证模式：v9 <model.pt> <image> —— 用真实yolov9-s权重验证GELAN实现
            if (args.Length >= 2 && System.IO.File.Exists(args[0]))
            {
                RunV9(args[0], args[1]);
                return;
            }

            // Assets shared with YoloSharpDemo (relative path: net48→Debug→bin→project→repo root)
            string assetRoot = System.IO.Path.GetFullPath(System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, @"..\..\..\..\YoloSharpDemo\Assets"));
            string preTrainedModelPath = System.IO.Path.Combine(assetRoot, @"PreTrainedModels\yolov8n-obb.bin");
            string predictImagePath = System.IO.Path.Combine(assetRoot, @"TestImage\trucks.jpg");

            Console.WriteLine("=== YoloSharp .NET Framework 4.8 验证 ===");
            Console.WriteLine("CLR: " + Environment.Version);
            Console.WriteLine("模型: " + System.IO.Path.GetFileName(preTrainedModelPath));
            Console.WriteLine("图像: " + System.IO.Path.GetFileName(predictImagePath));

            Mat predictImage = Cv2.ImRead(predictImagePath);
            if (predictImage.Empty())
            {
                Console.WriteLine("FAIL: 测试图像读取失败 " + predictImagePath);
                return;
            }
            Console.WriteLine("OpenCvSharp OK, 图像尺寸: " + predictImage.Width + "x" + predictImage.Height);

            // 与原Demo一致的配置（CPU设备，避免驱动差异干扰验证）
            Config config = new Config
            {
                DeviceType = DeviceType.CPU,
                ScalarType = ScalarType.BFloat16,
                YoloType = YoloType.Yolov8,
                YoloSize = YoloSize.n,
                TaskType = TaskType.Obb,
                ImageSize = 640,
                NumberClass = 15,
            };

            YoloTask yoloTask = new YoloTask(config);

            // 加载bin模型（TorchSharp加载权重 → 验证libtorch原生链）
            yoloTask.LoadModel(preTrainedModelPath);

            // CPU推理
            System.Diagnostics.Stopwatch sw = System.Diagnostics.Stopwatch.StartNew();
            List<YoloResult> predictResult = yoloTask.ImagePredict(predictImage);
            sw.Stop();

            Console.WriteLine();
            Console.WriteLine("推理结果: " + predictResult.Count + " 个目标, 耗时 " + sw.ElapsedMilliseconds + " ms");

            // 输出结果（不画图，聚焦链路验证；画图部分与原Demo相同逻辑）
            foreach (YoloResult result in predictResult)
            {
                Console.WriteLine(string.Format("LabelID:{0}, Score:{1:F1}%, CX:{2:F0}, CY:{3:F0}, W:{4:F0}, H:{5:F0}, R:{6:F3}",
                    result.ClassID, result.Score * 100, result.CenterX, result.CenterY,
                    result.Width, result.Height, result.Radian));
            }

            Console.WriteLine();
            Console.WriteLine("=== PASS: net48 + TorchSharp(netstandard2.0) + YoloSharp 推理链路可用 ===");
        }

        /// <summary>YOLOv9(GELAN)验证：加载真实.pt权重（PtReader）→推理→对照Python结果</summary>
        static void RunV9(string modelPath, string imagePath)
        {
            Console.WriteLine("=== YOLOv9 GELAN 验证 ===");
            Console.WriteLine("模型: " + modelPath);
            Console.WriteLine("图像: " + imagePath);

            Config config = new Config
            {
                DeviceType = DeviceType.CPU,     // CPU先验证链路（与主验证一致避免驱动差异）
                ScalarType = ScalarType.BFloat16,
                YoloType = YoloType.Yolov9,
                YoloSize = YoloSize.s,
                TaskType = TaskType.Detection,
                ImageSize = 640,
                NumberClass = 1,
            };

            YoloTask yoloTask = new YoloTask(config);
            yoloTask.LoadModel(modelPath);

            Mat predictImage = Cv2.ImRead(imagePath);
            if (predictImage.Empty()) { Console.WriteLine("FAIL: 图像读取失败"); return; }
            Console.WriteLine("图像: " + predictImage.Width + "x" + predictImage.Height);

            var sw = System.Diagnostics.Stopwatch.StartNew();
            List<YoloResult> results = yoloTask.ImagePredict(predictImage);
            sw.Stop();
            Console.WriteLine($"推理结果: {results.Count} 个目标, 耗时 {sw.ElapsedMilliseconds}ms");
            foreach (var r in results)
                Console.WriteLine($"ClassID:{r.ClassID} Score:{r.Score * 100:F1}% CX:{r.CenterX:F0} CY:{r.CenterY:F0} W:{r.Width:F0} H:{r.Height:F0}");
            Console.WriteLine(results.Count > 0 ? "=== v9 PASS ===" : "=== v9 FAIL(0目标) ===");
        }
    }
}
