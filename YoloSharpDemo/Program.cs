using Data;
using OpenCvSharp;
using YoloSharp.Models;
using YoloSharp.Types;

namespace YoloSharpDemo
{
    internal class Program
    {
        /// <summary>无效任务×架构组合→弹窗说明原因（Windows），返回true=无效已提示</summary>
        static bool WarnInvalidCombo(YoloType yolo, TaskType task)
        {
            string? why = (yolo, task) switch
            {
                (YoloType.Yolov9, TaskType.Obb) =>
                    "官方 YOLOv9 (GELAN) 从未发布 OBB（旋转框）变体。\n" +
                    "→ 请改用 Yolov8 / Yolov11 训练 OBB，或把 TaskType 改为 Detection。",
                (YoloType.Yolov9, TaskType.Pose) =>
                    "官方 YOLOv9 (GELAN) 从未发布 Pose（姿态估计）变体。\n" +
                    "→ 请改用 Yolov8 / Yolov11 训练 Pose，或把 TaskType 改为 Detection。",
                (YoloType.Yolov9, TaskType.Classification) =>
                    "官方 YOLOv9 存在分类变体（yolov9-cls），但本 C# 移植版尚未实现。\n" +
                    "→ 请暂用 Yolov8 / Yolov11 训练分类；v9-cls 头欢迎社区补充。",
                _ => null,
            };
            if (why == null) return false;

            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("========================================");
            Console.WriteLine($"ERROR 不支持的组合: {yolo} + {task}");
            Console.WriteLine(why);
            Console.WriteLine("========================================");
            Console.ResetColor();
            return true;
        }

        static void Main(string[] args)
        {
            // 统一入口（v8/v9/v11/v12同一流程，仅Config不同）：
            //   YoloSharpDemo [yoloType] [task] [predictImgsz] [predictConf] [epochs]
            //   yoloType:     Yolov5u/Yolov8/Yolov9/Yolov11/Yolov12（默认Yolov8）
            //   task:         Detection/Obb/Segmentation/Pose/Classification（默认Obb——沿用示例数据集dotav1）
            //   predictImgsz: 推理输入尺寸（letterbox等比缩放目标，需32倍数，默认640）
            //                 ——应与训练尺寸一致；训练1280的模型推理传1280，否则精度下降
            //   predictConf:  置信度阈值（默认0.45）
            //   epochs:       训练轮数（默认10；dotav1-obb/v9示例均适用）
            // 例：
            //   YoloSharpDemo                              → v8+Obb 默认（训练+推理）
            //   YoloSharpDemo Yolov9 Detection             → v9 从头训练dotav1+推理
            //   YoloSharpDemo Yolov8 Obb 1280 0.3          → v8 训练后按1280推理、conf=0.3
            //   YoloSharpDemo Yolov9 Segmentation          → v9-seg（需自备权重路径见下）
            YoloType yoloType = args.Length > 0 ? (YoloType)Enum.Parse(typeof(YoloType), args[0], true) : YoloType.Yolov8;
            TaskType taskType = args.Length > 1 ? (TaskType)Enum.Parse(typeof(TaskType), args[1], true) : TaskType.Obb;
            int predictImgsz = args.Length > 2 ? int.Parse(args[2]) : 0;   // 0=用config.ImageSize
            float predictConf = args.Length > 3 ? float.Parse(args[3]) : 0.45f;
            int epochs = args.Length > 4 ? int.Parse(args[4]) : 10;

            string preTrainedModelPath = @"..\..\..\Assets\PreTrainedModels\yolov8n-obb.pt"; // Pretrained model path.
            string predictImagePath = @"..\..\..\Assets\TestImage\trucks.jpg";

            string dataRootPath = @"..\..\..\Assets\DataSets\dotav1";

            string trainDataPath = @"train.txt";
            string valDataPath = @"val.txt";

            Mat predictImage = Cv2.ImRead(predictImagePath);

            // Create a Yolo config（任务相关参数按task自动适配；v9等无预训练权重的架构从头训练）
            // nc: 示例数据集dotav1为15类（train.txt/val.txt标签0..14）；Classification任务无框标签按1类
            int nc = taskType == TaskType.Classification ? 1 : 15;
            Config config = new Config
            {
                DeviceType = DeviceType.CUDA,
                ScalarType = ScalarType.BFloat16,
                RootPath = dataRootPath,
                TrainDataPath = trainDataPath,
                ValDataPath = valDataPath,
                YoloType = yoloType,
                YoloSize = yoloType == YoloType.Yolov9 ? YoloSize.s : YoloSize.n,   // v9官方最小为s
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

            // Create a yolo task. (invalid task×arch combos are shown a warning popup here)
            if (WarnInvalidCombo(config.YoloType, config.TaskType)) return;
            YoloTask yoloTask = new YoloTask(config);

            // Load pre-trained model (skip for architectures without one in Assets — trains from scratch).
            if (System.IO.File.Exists(preTrainedModelPath))
                yoloTask.LoadModel(preTrainedModelPath, skipNcNotEqualLayers: true);

            // Train model
            yoloTask.Train();

            // Predict image, if the model is not trained or loaded, it will use random weight to predict.
            // [predict带参] imgsz显式传入时覆盖config（训练1280的模型推理传1280）；conf可调阈值
            if (predictImgsz >= 32) config.ImageSize = predictImgsz - predictImgsz % 32;
            Console.WriteLine($"Predict {yoloType}+{taskType} imgsz={config.ImageSize} conf={predictConf}");
            List<YoloResult> predictResult = yoloTask.ImagePredict(predictImage, predictConf, 0.5f);

            // Rand for mask color.
            Random rand = new Random(1024);

            // Draw results
            foreach (YoloResult result in predictResult)
            {
                float[] cxcywhr = new float[] { result.CenterX, result.CenterY, result.Width, result.Height, result.Radian };
                float[] points = cxcywhr2xyxyxyxy(cxcywhr);

                OpenCvSharp.Point[] pts = new OpenCvSharp.Point[4]
                {
                    new OpenCvSharp.Point(points[0], points[1]),
                    new OpenCvSharp.Point(points[2], points[3]),
                    new OpenCvSharp.Point(points[4], points[5]),
                    new OpenCvSharp.Point(points[6], points[7]),
                };
                Cv2.Polylines(predictImage, new OpenCvSharp.Point[][] { pts }, true, OpenCvSharp.Scalar.Red, 2);
                string label = string.Format("{0}:{1:F1}%", result.ClassID, result.Score * 100);

                OpenCvSharp.Size textSize = Cv2.GetTextSize(label, HersheyFonts.HersheySimplex, 0.5, 1, out int baseline);
                Cv2.Rectangle(predictImage, new Rect(new OpenCvSharp.Point(result.X, result.Y - textSize.Height - baseline), new OpenCvSharp.Size(textSize.Width, textSize.Height + baseline)), OpenCvSharp.Scalar.White, Cv2.FILLED);
                Cv2.PutText(predictImage, label, new OpenCvSharp.Point(result.X, result.Y - baseline), HersheyFonts.HersheySimplex, 0.5, OpenCvSharp.Scalar.Black, 1);
                Console.WriteLine(string.Format("LabelID:{0}, Score:{1:F1}%, CenterX:{2}, CenterY:{3}, Width:{4}, Height:{5}, R:{6:F3} rnd", result.ClassID, result.Score * 100, result.CenterX, result.CenterY, result.Width, result.Height, result.Radian));

                // Draw mask
                if (result.Mask is not null)
                {
                    Mat maskMat = Mat.FromArray<byte>(result.Mask);
                    maskMat = maskMat * 255;
                    maskMat = maskMat.Transpose();

                    // Create random color
                    int R = rand.Next(0, 255);
                    int G = rand.Next(0, 255);
                    int B = rand.Next(0, 255);
                    OpenCvSharp.Scalar color = new OpenCvSharp.Scalar(R, G, B, 200);
                    Mat backColor = new Mat(maskMat.Rows, maskMat.Cols, MatType.CV_8UC3, color);
                    Cv2.Add(predictImage, backColor, predictImage, maskMat);
                }

                if (result.KeyPoints is not null)
                {
                    foreach (YoloSharp.Types.KeyPoint keyPoint in result.KeyPoints)
                    {
                        Cv2.Circle(predictImage, (int)keyPoint.X, (int)keyPoint.Y, 3, OpenCvSharp.Scalar.Blue);
                    }
                }
            }
            predictImage.SaveImage("result.jpg");

            Console.WriteLine();
            Console.WriteLine("Image Predict done");
        }

        /// <summary>
        /// CenterX, CenterY, Width, Height, r → P0(x0, y0), P1(x1, y1), P2(x2, y2), P3(x3, y3)
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        private static float[] cxcywhr2xyxyxyxy(float[] x)
        {
            float cx = x[0];
            float cy = x[1];
            float w = x[2];
            float h = x[3];
            float r = x[4];
            float cosR = (float)Math.Cos(r);
            float sinR = (float)Math.Sin(r);
            float wHalf = w / 2;
            float hHalf = h / 2;
            return new float[]
            {
                cx - wHalf * cosR + hHalf * sinR,
                cy - wHalf * sinR - hHalf * cosR,
                cx + wHalf * cosR + hHalf * sinR,
                cy + wHalf * sinR - hHalf * cosR,
                cx + wHalf * cosR - hHalf * sinR,
                cy + wHalf * sinR + hHalf * cosR,
                cx - wHalf * cosR - hHalf * sinR,
                cy - wHalf * sinR + hHalf * cosR,
            };
        }

        /// <summary>YOLOv9训练验证：GELAN骨架从头训练（Detection）dotav1→推理trucks.jpg</summary>
    }
}

