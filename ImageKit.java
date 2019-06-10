package image_procs;

import java.awt.FlowLayout;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import net.coobird.thumbnailator.Thumbnails;

public class ImageKit {

	// given
	protected Mat img;
	protected double fold; // image display fold
	protected int peakRange; // coverage range of histogram peak (suggested 7)

	// derived
	protected List<Mat> bgr; // R/G/B channel of img
	protected Mat norm; // normalised img
	protected Histogram[] hists; // R/G/B channel histogram for img
	protected int[] repreValue; // representative RGB value for img

	public ImageKit(Mat img, double fold) {
		this.img = img;
		this.fold = fold;
	}

	public ImageKit(Mat img, double fold, int peakRange) {
		this.img = img;
		this.fold = fold;
		this.peakRange = peakRange;

		bgr = new ArrayList<>();
		hists = new Histogram[3];
		Core.split(img, bgr);
		hist();
		normalization();
	}

	// Image display kit
	public static BufferedImage resize(BufferedImage img, int newW, int newH) throws IOException {
		return Thumbnails.of(img).size(newW, newH).asBufferedImage();
	}

	public BufferedImage Mat2BufferedImage(Mat m) {
		// source:
		// http://answers.opencv.org/question/10344/opencv-java-load-image-to-gui/
		// Fastest code
		// The output can be assigned either to a BufferedImage or to an Image

		int type = BufferedImage.TYPE_BYTE_GRAY;
		if (m.channels() > 1)
			type = BufferedImage.TYPE_3BYTE_BGR;
		int bufferSize = m.channels() * m.cols() * m.rows();
		byte[] b = new byte[bufferSize];
		m.get(0, 0, b); // get all the pixels
		BufferedImage image = new BufferedImage(m.cols(), m.rows(), type);
		final byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
		System.arraycopy(b, 0, targetPixels, 0, b.length);
		return image;
	}

	public void displayImage(Image img2) {
		ImageIcon icon = new ImageIcon(img2);
		JFrame frame = new JFrame();
		frame.setLayout(new FlowLayout());
		frame.setSize(img2.getWidth(null), img2.getHeight(null));
		JLabel lbl = new JLabel();
		lbl.setIcon(icon);
		frame.add(lbl);
		frame.setVisible(true);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	}

	public void show(Mat img) throws IOException {
		BufferedImage bimg = Mat2BufferedImage(img);
		int newW = (int) (bimg.getWidth() * fold);
		int newH = (int) (bimg.getHeight() * fold);
		BufferedImage rsz = resize(bimg, newW, newH);
		displayImage(rsz);
	}

	// Histogram
	public void hist() {
		for (int i = 0; i < 2; i++) {
			hists[i] = new Histogram(bgr.get(i), peakRange);
			repreValue[i] = hists[i].minorPeakInten;
		}
	}

	// Normalisation - histogram based
	public void normalization() {
		int h = (int) img.size().height, w = (int) img.size().width;
		norm = new Mat(h, w, CvType.CV_8UC3, new Scalar(0, 0, 0));
		for (int i = 0; i < 3; i++) {
			Mat nChl = new Mat(), mask = new Mat();
			// shift current channel
			int shift = hists[i].majorPeakInten - 200;
			Mat smask = new Mat(h, w, CvType.CV_8UC1, new Scalar(shift));

			// do not shift 0-valued pixels
			Imgproc.threshold(bgr.get(i), mask, 0, 1, 0); // pixels originally > 0 are converted to 1
			Core.multiply(mask, smask, mask); // pixels originally > 0 assigned with shift

			// add
			Core.add(mask, bgr.get(i), nChl);
			Core.insertChannel(nChl, norm, i);
		}
	}

	public static void main(String[] args) throws IOException {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		String nm = "D:/Software/Python37/wd/2018-19FYP/chip_v2/Benchmarking/20190305/version2/1-orig/light3/t1l3_20190305_18_lighttray_900lux_1min_TET0.5.jpg";
		int[] size = { 7, 6, 20, 27, 40, 30, 35, 30, 15, 420 };
		Chip chip = new Chip(nm, size);
		Mat b5 = chip.blocks[5];
		ImageKit ik = new ImageKit(b5, 1, 7);
		Mat b5ro = new Mat(), b5rn = new Mat();
		Core.extractChannel(b5, b5ro, 0);
		Core.extractChannel(ik.norm, b5rn, 0);

		// check border remains 0
		for (int i = 0; i < 40; i++) {
			for (int j = 60; j < 70; j++) {
				double[] cur = b5ro.get(i, j);
				double[] curn = b5rn.get(i, j);
				System.out.print((int) cur[0] + ",");
				System.out.print("(" + (int) curn[0] + ")");
			}
			System.out.println();
		}
	}
}