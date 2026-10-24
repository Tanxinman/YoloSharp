using Data;
using OpenCvSharp;
using SkiaSharp;
using TorchSharp;
using YoloSharp.Types;
using YoloSharp.Utils;

namespace YoloSharp.Models
{
    public class YoloTask
    {
        private readonly YoloBaseTaskModel yolo;

        private bool Initialized => yolo != null;

        public YoloTask(Config config)
        {
            // Fail fast on invalid task x architecture combinations (e.g. Yolov9+Obb):
            // reject with a clear message instead of a generic exception deep inside a task-model ctor.
            ValidateTaskYoloCombo(config.TaskType, config.YoloType);

            if (string.IsNullOrEmpty(config.OutputPath))
            {
                config.OutputPath = Path.Combine("result", config.TaskType.ToString().ToUpper(), DateTime.Now.ToString("yyyyMMddHHmmss"));
            }
            torchvision.io.DefaultImager = new torchvision.io.SkiaImager();
            yolo = config.TaskType switch
            {
                TaskType.Detection => new Detector(config),
                TaskType.Segmentation => new Segmenter(config),
                TaskType.Obb => new Obber(config),
                TaskType.Pose => new PoseDetector(config),
                TaskType.Classification => new Classifier(config),
                _ => throw new NotImplementedException("Task type not support now.")
            };
        }

        /// <summary>
        /// Fail-fast validation of the task x architecture support matrix.
        /// YOLOv9 (GELAN) 官方支持矩阵（WongKinYiu/yolov9 + ultralytics docs）：
        ///   Detection ✔（本C#实现已贡献）
        ///   Segmentation/Classification：官方存在，本C#实现尚未贡献（报错注明，欢迎社区补充）
        ///   Pose/OBB：官方就不存在这两个变体
        /// </summary>
        static void ValidateTaskYoloCombo(TaskType task, YoloType yolo)
        {
            if (yolo != YoloType.Yolov9) return;

            switch (task)
            {
                case TaskType.Detection:
                    return;   // 已实现
                case TaskType.Segmentation:
                    return;   // 已实现（Yolov9Segment, gelan-c-seg）
                case TaskType.Classification:
                    throw new NotSupportedException(
                        $"{yolo} + {task}: official YOLOv9 ships this variant, but the C# port has not implemented it yet. " +
                        "Use Yolov8/Yolov11 for this task, or contribute the v9 {task} head.");
                case TaskType.Obb:
                case TaskType.Pose:
                    throw new NotSupportedException(
                        $"Invalid combination: {yolo} + {task}. " +
                        "Official YOLOv9 (GELAN) does not ship OBB/Pose variants at all. " +
                        "Use Yolov8/Yolov11 for this task.");
                default:
                    return;
            }
        }

        public void LoadModel(string path, bool skipNcNotEqualLayers = false)
        {
            if (!Initialized)
            {
                throw new ArgumentNullException("Yolo is not Initialized.");
            }
            yolo?.LoadModel(path, skipNcNotEqualLayers);
        }

        public void Train()
        {
            if (!Initialized)
            {
                throw new ArgumentNullException("Yolo is not Initialized.");
            }
            yolo?.Train();
        }

        public List<YoloResult> ImagePredict(torch.Tensor orgImage)
        {
            if (!Initialized)
            {
                throw new ArgumentNullException("Yolo is not Initialized.");
            }
            return yolo.ImagePredict(orgImage);
        }

        public List<YoloResult> ImagePredict(SKBitmap image)
        {
            torch.Tensor orgImage = Lib.GetTensorFromImage(image);
            return ImagePredict(orgImage);
        }

        public List<YoloResult> ImagePredict(string imagePath)
        {
            torch.Tensor orgImage = Lib.GetTensorFromImage(imagePath);
            return ImagePredict(orgImage);
        }

        public List<YoloResult> ImagePredict(Mat mat)
        {
            torch.Tensor orgImage = Lib.GetTensorFromImage(mat);
            return ImagePredict(orgImage);
        }

        public List<YoloResult> ImagePredict(torch.Tensor orgImage, float predictThreshold, float iouThreshold)
        {
            if (!Initialized)
            {
                throw new ArgumentNullException("Yolo is not Initialized.");
            }
            return yolo.ImagePredict(orgImage, predictThreshold, iouThreshold);
        }

        public List<YoloResult> ImagePredict(SKBitmap image, float predictThreshold, float iouThreshold)
        {
            torch.Tensor orgImage = Lib.GetTensorFromImage(image);
            return ImagePredict(orgImage, predictThreshold, iouThreshold);
        }

        public List<YoloResult> ImagePredict(string imagePath, float predictThreshold, float iouThreshold)
        {
            torch.Tensor orgImage = Lib.GetTensorFromImage(imagePath);
            return ImagePredict(orgImage, predictThreshold, iouThreshold);
        }

        public List<YoloResult> ImagePredict(Mat mat, float predictThreshold, float iouThreshold)
        {
            torch.Tensor orgImage = Lib.GetTensorFromImage(mat);
            return ImagePredict(orgImage, predictThreshold, iouThreshold);
        }


    }
}
