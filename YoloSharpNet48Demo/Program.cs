using System;
using System.Collections.Generic;
using System.IO;
using Data;
using OpenCvSharp;
using YoloSharp.Models;
using YoloSharp.Types;

namespace YoloSharpNet48Demo
{
    /// <summary>
    /// YoloSharp net48 兼容性验证：
    /// 与 YoloSharpDemo（net6.0）相同的功能逻辑，验证 .NET Framework 4.8 下
    /// TorchSharp + libtorch + OpenCvSharp + YoloSharp(netstandard2.0) 训练+推理链路。
    /// 统一入口（v5u/v8/v9/v11/v12 同一流程）：
    ///   YoloSharpNet48Demo [yoloType] [task] [predictImgsz] [predictConf] [epochs]
    /// </summary>
    internal class Program
    {
        static void Main(string[] args)
        {
            // 统一入口（与YoloSharpDemo一致）：
            //   yoloType:     Yolov5u/Yolov8/Yolov9/Yolov11/Yolov12（默认Yolov8）
            //   task:         Detection/Obb/Segmentation/Pose/Classification（默认Obb——沿用示例数据集dotav1）
            //   predictImgsz: 推理输入尺寸（letterbox等比缩放目标，需32倍数，默认640）
            //                 ——应与训练尺寸一致；训练1280的模型推理传1280，否则精度下降
            //   predictConf:  置信度阈值（默认0.45）
            //   epochs:       训练轮数（默认10；net48下训练较慢建议1-2轮验证链路）
            YoloType yoloType = args.Length > 0 ? (YoloType)Enum.Parse(typeof(YoloType), args[0], true) : YoloType.Yolov8;
            TaskType taskType = args.Length > 1 ? (TaskType)Enum.Parse(typeof(TaskType), args[1], true) : TaskType.Obb;
            int predictImgsz = args.Length > 2 ? int.Parse(args[2]) : 0;   // 0=用config.ImageSize
            float predictConf = args.Length > 3 ? float.Parse(args[3]) : 0.45f;
            int epochs = args.Length > 4 ? int.Parse(args[4]) : 10;

            // Assets shared with YoloSharpDemo (relative path: net48→Debug→bin→project→repo root)
            string assetRoot = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, @"..\..\..\..\YoloSharpDemo\Assets"));
            string preTrainedModelPath = Path.Combine(assetRoot, @"PreTrainedModels\yolov8n-obb.pt");
            string predictImagePath = Path.Combine(assetRoot, @"TestImage\trucks.jpg");
            string dataRootPath = Path.Combine(assetRoot, @"DataSets\DOTAv1");
            string trainDataPath = "train.txt";
            string valDataPath = "val.txt";

            Console.WriteLine("=== YoloSharp .NET Framework 4.8 验证 ===");
            Console.WriteLine("CLR: " + Environment.Version);
            Console.WriteLine("组合: " + yoloType + "+" + taskType + " epochs=" + epochs);

            Mat predictImage = Cv2.ImRead(predictImagePath);
            if (predictImage.Empty())
            {
                Console.WriteLine("FAIL: 测试图像读取失败 " + predictImagePath);
                return;
            }
            Console.WriteLine("OpenCvSharp OK, 图像尺寸: " + predictImage.Width + "x" + predictImage.Height);

            // Create a Yolo config（与YoloSharpDemo同规则：nc跟数据集dotav1=15类；Classification按1类）
            int nc = taskType == TaskType.Classification ? 1 : 15;
            Config config = new Config
            {
                DeviceType = DeviceType.CUDA,       // 与net6.0版一致走CUDA（net48验证含GPU链路）
                ScalarType = ScalarType.BFloat16,
                RootPath = dataRootPath,
                TrainDataPath = trainDataPath,
                ValDataPath = valDataPath,
                YoloType = yoloType,
                YoloSize = yoloType == YoloType.Yolov9 ? YoloSize.s : YoloSize.n,
                TaskType = taskType,
                ImageSize = 640,
                BatchSize = 8,
                NumberClass = nc,
                Workers = 4,
                Epochs = epochs,
                LearningRate = 2e-4f,
                KeyPoint_Num = 17,
                KeyPoint_Dim = 3,
                CloseMosaic = 10,
            };

            // Create a yolo task. (invalid task×arch combos are shown a warning here)
            if (WarnInvalidCombo(config.YoloType, config.TaskType)) return;
            YoloTask yoloTask = new YoloTask(config);

            // Load pre-trained model (skip for architectures without one in Assets — trains from scratch).
            if (File.Exists(preTrainedModelPath))
                yoloTask.LoadModel(preTrainedModelPath, skipNcNotEqualLayers: true);

            // Train model
            yoloTask.Train();

            // Predict image, if the model is not trained or loaded, it will use random weight to predict.
            // [predict带参] imgsz显式传入时覆盖config（训练1280的模型推理传1280）；conf可调阈值
            if (predictImgsz >= 32) config.ImageSize = predictImgsz - predictImgsz % 32;
            Console.WriteLine("Predict " + yoloType + "+" + taskType + " imgsz=" + config.ImageSize + " conf=" + predictConf);

            System.Diagnostics.Stopwatch sw = System.Diagnostics.Stopwatch.StartNew();
            List<YoloResult> predictResult = yoloTask.ImagePredict(predictImage, predictConf, 0.5f);
            sw.Stop();

            Console.WriteLine();
            Console.WriteLine("推理结果: " + predictResult.Count + " 个目标, 耗时 " + sw.ElapsedMilliseconds + " ms");

            // 输出结果（不画图，聚焦链路验证；画图部分与net6.0版相同逻辑可自行补）
            foreach (YoloResult result in predictResult)
            {
                Console.WriteLine(string.Format("LabelID:{0}, Score:{1:F1}%, CX:{2:F0}, CY:{3:F0}, W:{4:F0}, H:{5:F0}, R:{6:F3}",
                    result.ClassID, result.Score * 100, result.CenterX, result.CenterY,
                    result.Width, result.Height, result.Radian));
            }

            Console.WriteLine();
            Console.WriteLine("=== PASS: net48 + TorchSharp(netstandard2.0) + YoloSharp 训练+推理链路可用 ===");
        }

        /// <summary>非法 架构×任务 组合拦截（与YoloSharpDemo同款）</summary>
        private static bool WarnInvalidCombo(YoloType yolo, TaskType task)
        {
            string why = null;
            if (yolo == YoloType.Yolov9 && task == TaskType.Pose)
                why = "官方 YOLOv9 (GELAN) 从未发布 Pose（姿态估计）变体。\n→ 请改用 Yolov8 / Yolov11 训练 Pose，或把 TaskType 改为 Detection。";
            else if (yolo == YoloType.Yolov9 && task == TaskType.Classification)
                why = "官方 YOLOv9 从未发布 Classification 变体。\n→ 请改用 Yolov8 / Yolov11，或把 TaskType 改为 Detection/Segmentation。";
            if (why == null) return false;

            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("========================================");
            Console.WriteLine("ERROR 不支持的组合: " + yolo + " + " + task);
            Console.WriteLine(why);
            Console.WriteLine("========================================");
            Console.ResetColor();
            return true;
        }
    }
}
