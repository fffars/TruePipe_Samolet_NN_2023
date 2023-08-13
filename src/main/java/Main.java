import org.opencv.core.*;
import org.opencv.objdetect.CascadeClassifier;
import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
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
                BufferedImage img = ImageIO.read(new File(args[0]));
//                    BufferedImage resizedImage = res  izeImage(img, 640, 480);
                Graphics2D g2d = img.createGraphics();
                g2d.setColor(Color.RED);
                g2d.setStroke(new BasicStroke(5));
                CatImageDetector detecor = new CatImageDetector();
                detecor.work_with_image(img);
                Rect[] array = detecor.faces.toArray();
                for (Rect rect : array) {
                    g2d.drawRect(rect.x, rect.y, rect.width, rect.height);
                }
                g2d.setColor(Color.GREEN);
                g2d.setStroke(new BasicStroke(5));
                Rect[] array2 = detecor.faces2.toArray();
                for (Rect rect : array2) {
                    g2d.drawRect(rect.x, rect.y, rect.width, rect.height);
                }
                g2d.dispose();
                ImageIcon icon = new ImageIcon(img);
                JFrame window = new JFrame("Результат");
                window.setLocationByPlatform(true);
                window.setPreferredSize(new Dimension(img.getWidth(), img.getHeight()));
                window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                JLabel lbl = new JLabel();
                lbl.setIcon(icon);
                window.add(lbl);
                window.getContentPane().add(lbl, BorderLayout.CENTER);
                window.pack();
                window.setVisible(true);
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

    public int[] work_with_image(BufferedImage img) {
        int[] rez = {0, 0};
        CascadeClassifier face_cascade = new CascadeClassifier("haarcascade_frontalcatface.xml");
        if (face_cascade.empty()) {
            System.out.println("Ошибка загрузки");
            return rez;
        } else {
            System.out.println("Загрузили \"haarcascade_frontalcatface\"");
        }

        byte[] pixels = ((DataBufferByte) img.getRaster().getDataBuffer()).getData();
        Mat inputFrame = new Mat(img.getHeight(), img.getWidth(), CvType.CV_8UC3);
        inputFrame.put(0, 0, pixels);

        face_cascade.detectMultiScale(inputFrame, faces);
        rez[0] = faces.toArray().length;

        System.out.println("Обнаружено " + rez[0] + " котов");

        face_cascade = new CascadeClassifier("haarcascade_frontalcatface_extended.xml");
        if (face_cascade.empty()) {
            System.out.println("Ошибка загрузки");
            return rez;
        } else {
            System.out.println("Загрузили \"haarcascade_frontalcatface_extended\"");
        }

        face_cascade.detectMultiScale(inputFrame, faces2);
        rez[1] = faces2.toArray().length;

        System.out.println("Обнаружено " + rez[1] + " котов");
        return rez;
    }
}