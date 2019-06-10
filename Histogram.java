package image_procs;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;

public class Histogram {

	// given
	protected Mat imgChl; // a particular channel of image
	protected int dist;

	// derived
	protected int[] hist, curChannel;
	protected int majorPeakInten; // intensity value attaining the major peak
	protected int minorPeakInten; // minor peak
	protected List<Integer> mrPeaks; // candidate minor peaks

	public Histogram(Mat img, int dist) {
		imgChl = img;
		this.dist = dist;

		MatToIntArray(); // update curChannel
		hist = new int[256];
		for (int val : curChannel)
			hist[val]++;
		hist[0] = 0;
		hist[255] = 0;
		peak();
	}

	public void MatToIntArray() {
		curChannel = new int[(int) (imgChl.total() * imgChl.channels())];
		MatOfInt moi = new MatOfInt(CvType.CV_32S);
		imgChl.convertTo(moi, CvType.CV_32S);
		moi.get(0, 0, curChannel);
	}

	/**
	 * Recursively squeeze to disclose a set of candidate minor peaks
	 * 
	 * @return a set of candidate minor peaks
	 */
	private void mrPeak() {
		mrPeaks = new ArrayList<Integer>();
		for (int i = dist; i < majorPeakInten - dist; i++) {
			if (hist[i] < hist[i - 1] || hist[i] < hist[i + 1])
				continue;
			int lo = i, hi = i, count = dist;
			// leftward check
			while (--lo >= 0 && count > 0) {
				if (hist[lo] > hist[i])
					break;
				if (hist[lo] < hist[i])
					count--;
			}
			if (count != 0)
				continue;
			// rightward check
			count = dist;
			while (++hi < 256 && count > 0) {
				if (hist[hi] > hist[i])
					break;
				if (hist[hi] < hist[i])
					count--;
			}
			if (count != 0)
				i = hi - 1;
			else
				mrPeaks.add(i); // candidate found
		}
	}

	/**
	 * Detect major and minor peak update majorPeak & minorPeak
	 */
	public void peak() {
		// major peak detection - bin with largest count
		for (int i = 0, maxCount = -1; i < 256; i++) {
			if (hist[i] <= maxCount)
				continue;
			maxCount = hist[i];
			majorPeakInten = i;
		}
		// minor peak - largest among candidates
		int large = -1;
		mrPeak(); // update mrPeaks
		for (Integer postuR : mrPeaks) {
			if (hist[postuR] <= large)
				continue;
			large = hist[postuR];
			minorPeakInten = postuR;
		}
	}

	public static void main(String[] args) throws IOException {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		String nm = "D:/Software/Python37/wd/2018-19FYP/chip_v2/Benchmarking/20190305/version2/1-orig/light3/t1l3_20190305_18_lighttray_900lux_1min_TET0.5.jpg";
		int[] size = { 7, 6, 20, 27, 40, 30, 35, 30, 15, 420 };
		Chip chip = new Chip(nm, size);

		Mat b5 = chip.blocks[5], b5r = new Mat();
		Core.extractChannel(b5, b5r, 0);
		Histogram hist = new Histogram(b5r, 7);
		for (int i = 0; i < 256; i++) {
			int cur = hist.hist[i];
			System.out.print(Integer.toString(i) + ": " + Integer.toString(cur) + "| ");
			if (i % 10 == 0)
				System.out.println();
		}
		System.out.println(hist.majorPeakInten + " " + hist.minorPeakInten);
		List<Integer> res = hist.mrPeaks;
		for (Integer i : res)
			System.out.print(Integer.toString(i) + ", ");
	}

}