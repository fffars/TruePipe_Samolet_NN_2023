import org.opencv.core.*;
import org.opencv.dnn.DetectionModel;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.FaceDetectorYN;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.DataBufferInt;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.LinkedList;
import java.util.function.Consumer;

import static org.opencv.core.CvType.CV_32F;
//import org.tensorflow.ConcreteFunction;
//import org.tensorflow.Signature;
//import org.tensorflow.Tensor;
//import org.tensorflow.TensorFlow;
//import org.tensorflow.op.Ops;
//import org.tensorflow.op.core.Placeholder;
//import org.tensorflow.op.math.Add;
//import org.tensorflow.types.TInt32;

public class Main {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // вывести в окне картинку
        // нарисовать рамки в области распознавания
        // запуск по ключу
        SwingUtilities.invokeLater(() -> {
            try {
                BufferedImage img_origin = ImageIO.read(new File(args[0]));
//                    BufferedImage resizedImage = res  izeImage(img, 640, 480);
                Image tmp = img_origin.getScaledInstance(640, 480, Image.SCALE_SMOOTH);
                BufferedImage dimg = new BufferedImage(640, 480, BufferedImage.TYPE_4BYTE_ABGR);
                Graphics2D g2d2 = dimg.createGraphics();
                g2d2.drawImage(tmp, 0, 0, null);
                g2d2.dispose();
                CatImageDetector detecor = new CatImageDetector();
                detecor.work_with_image(dimg);
                if (args.length > 1 && args[1].equalsIgnoreCase("-g")) {
                    Graphics2D g2d = dimg.createGraphics();
                    g2d.setColor(Color.RED);
                    g2d.setStroke(new BasicStroke(5));

//                Rect[] array = detecor.faces.toArray();
//                for (Rect rect : array) {
//                    g2d.drawRect(rect.x, rect.y, rect.width, rect.height);
//                }
                    byte[] pixels = ((DataBufferByte) dimg.getRaster().getDataBuffer()).getData();
                    Mat inputFrame = new Mat(dimg.getHeight(), dimg.getWidth(), CvType.CV_8UC3);
                    inputFrame.put(0, 0, pixels);
                    Mat mat = Dnn.blobFromImage(inputFrame, 1, new Size(640, 640));
                    detecor.rezults.forEach(new Consumer<Mat>() {
                        @Override
                        public void accept(Mat mat) {
//                                                        mat.get()
                        }
                    });
//                    for (int i = 0; i < detecor.detect_rez.rows(); i++) {
////                    float confidence = detecor.detect_rez.at(Float.class, i, 2).getV();
//                        // Check if the detection is of good quality
//                        int x = (int) (detecor.detect_rez.at(Float.class, i, 3).getV() * mat.cols());
//                        int y = (int) (detecor.detect_rez.at(Float.class, i, 4).getV() * mat.rows());
//                        int width = (int) (detecor.detect_rez.at(Float.class, i, 5).getV() * mat.cols() - x);
//                        int height = (int) (detecor.detect_rez.at(Float.class, i, 6).getV() * mat.rows() - y);
//                        g2d.drawRect(x, y, width, height);
//
//                    }
//                g2d.setColor(Color.GREEN);
//                g2d.setStroke(new BasicStroke(5));
//                Rect[] array2 = detecor.faces2.toArray();
//                for (Rect rect : array2) {
//                    g2d.drawRect(rect.x, rect.y, rect.width, rect.height);
//                }
                    g2d.dispose();
                    ImageIcon icon = new ImageIcon(dimg);
                    JFrame window = new JFrame("Результат");
                    window.setLocationByPlatform(true);
                    window.setPreferredSize(new Dimension(dimg.getWidth(), dimg.getHeight()));
                    window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                    JLabel lbl = new JLabel();
                    lbl.setIcon(icon);
                    window.add(lbl);
                    window.getContentPane().add(lbl, BorderLayout.CENTER);
                    window.pack();
                    window.setVisible(true);
                }

            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });

        // использование TensorFlow
//        try (ConcreteFunction dbl = ConcreteFunction.create(Main::dbl);
//             TInt32 x = TInt32.scalarOf(10);
//             Tensor dblX = dbl.call(x)) {
//            System.out.println(x.getInt() + " doubled is " + ((TInt32)dblX).getInt());
//        }
    }

//    static BufferedImage resizeImage(BufferedImage originalImage, int targetWidth, int targetHeight) throws IOException {
//        BufferedImage resizedImage = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
//        Graphics2D graphics2D = resizedImage.createGraphics();
//        graphics2D.drawImage(originalImage, 0, 0, targetWidth, targetHeight, null);
//        graphics2D.dispose();
//        return resizedImage;
//    }

//    private static Signature dbl(Ops tf) {
//        Placeholder<TInt32> x = tf.placeholder(TInt32.class);
//        Add<TInt32> dblX = tf.math.add(x, x);
//        return Signature.builder().input("x", x).output("dbl", dblX).build();
//    }
}

class CatImageDetector {

    MatOfRect faces = new MatOfRect();
    MatOfRect faces2 = new MatOfRect();
    Mat detect_rez = new Mat();
    LinkedList<Mat> rezults;

    public int[] work_with_image(BufferedImage img) {

        int[] rez = {0, 0};
//        CascadeClassifier face_cascade = new CascadeClassifier("haarcascade_frontalcatface.xml");
//        if (face_cascade.empty()) {
//            System.out.println("Ошибка загрузки");
//            return rez;
//        } else {
//            System.out.println("Загрузили \"haarcascade_frontalcatface\"");
//        }

        Mat mat = matify(img);
//        byte[] pixels = ((DataBufferByte) img.getRaster().getDataBuffer()).getData();
//        Mat inputFrame = new Mat();
//        inputFrame.put(0, 0, pixels);
////        Net net = Dnn.readNetFromONNX("yolo8m_2000ph_100ep.onnx");
//        net.setInput(Dnn.blobFromImage(inputFrame, 1, new Size(640, 640)));
//        Mat mat2 = Dnn.blobFromImage(mat, 1, new Size(640, 640));
//        rezults = new LinkedList<>();
        // доделать нейросеть

        FaceDetectorYN faceDetectorYN = FaceDetectorYN.create("yolo8m_2000ph_100ep.onnx", "", new Size(640, 640));
        System.out.println("Модель \"yolo8m_2000ph_100ep.onnx\" загружена");
        faceDetectorYN.setInputSize(mat.size());
        faceDetectorYN.detect(mat, detect_rez);
//        det_model.detect(inputFrame, detect_rez, )
//        detect_rez = net.forward();

//        face_cascade.detectMultiScale(inputFrame, faces);
//        rez[0] = faces.toArray().length;

//        System.out.println("Обнаружено " + rez[0] + " котов");
//        int count = 0;
//        for (int i = 0; i < detect_rez.rows(); i++) {
//            if (detect_rez.at(Float.class, i, 2).getV() > 0.4) {
//                count++;
//            }
//        }
        System.out.println("Обнаружено " + detect_rez.rows() + " котов");

//        face_cascade = new CascadeClassifier("haarcascade_frontalcatface_extended.xml");
//        if (face_cascade.empty()) {
//            System.out.println("Ошибка загрузки");
//            return rez;
//        } else {
//            System.out.println("Загрузили \"haarcascade_frontalcatface_extended\"");
//        }
//
//        face_cascade.detectMultiScale(inputFrame, faces2);
//        rez[1] = faces2.toArray().length;
//
//        System.out.println("Обнаружено " + rez[1] + " котов");
        return rez;
    }

    public Mat matify(BufferedImage sourceImg) {

        java.awt.image.DataBuffer dataBuffer = sourceImg.getRaster().getDataBuffer();
        byte[] imgPixels = null;
        Mat imgMat = null;

        int width = sourceImg.getWidth();
        int height = sourceImg.getHeight();

        if (dataBuffer instanceof DataBufferByte) {
            imgPixels = ((DataBufferByte) dataBuffer).getData();
        }

        if (dataBuffer instanceof DataBufferInt) {

            int byteSize = width * height;
            imgPixels = new byte[byteSize * 3];

            int[] imgIntegerPixels = ((DataBufferInt) dataBuffer).getData();

            for (int p = 0; p < byteSize; p++) {
                imgPixels[p * 3 + 0] = (byte) ((imgIntegerPixels[p] & 0x00FF0000) >> 16);
                imgPixels[p * 3 + 1] = (byte) ((imgIntegerPixels[p] & 0x0000FF00) >> 8);
                imgPixels[p * 3 + 2] = (byte) (imgIntegerPixels[p] & 0x000000FF);
            }
        }

        if (imgPixels != null) {
            imgMat = new Mat(height, width, CvType.CV_8UC3);
            imgMat.put(0, 0, imgPixels);
        }

        return imgMat;
    }
}