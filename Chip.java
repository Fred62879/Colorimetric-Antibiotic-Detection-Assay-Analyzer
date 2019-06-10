package image_procs;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.opencv.core.*;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.imgproc.Imgproc;
import obsolete.PersTransformation;
import org.opencv.imgcodecs.Imgcodecs;

public class Chip {

	// given
	protected String filename;
	protected int[] size; // design data of the real chip (distances in 10 fold)
	/*
	 * size array components: 0/1 # of row/column; 2/3 horizontal/vertical distance
	 * between left/top_most point of the top-left spot and the left/top inner chip
	 * edge; 4/5 horizontal/vertical distance between two spots; 6/7 half of
	 * horizontal/vertical between two spot centres; 8 radius of spot 9 edge length
	 */

	// derived
	protected Mat orig;
	protected Mat gray;
	protected MatOfPoint inner; // inner contour of the black border surrounding the chip
	protected MatOfPoint2f mjCorners; // corners of major chip portion in clockwise order
	protected MatOfPoint2f mrCorners; // ... minor chip ...
	protected Mat restr; // finally transformed chip image
	protected Mat remediated; // remediated restr - residual border and glare removed
	protected Mat assem; // assembled from blocks array, should be the same as restr - correction check
	
	// block information
	protected Mat[] blocks; // all blocks - 42
	protected Mat[] normBlocks; // all normalised blocks - 42
	protected int[][] blockValue; // representative RGB value for each block - 42x3

	public Chip(String filename, int[] size) throws IOException {
		this.filename = filename;
		this.size = size;
		restr = new Mat();
		blocks = new Mat[size[0] * size[1]];
		normBlocks = new Mat[size[0] * size[1]];
		blockValue = new int[size[0] * size[1]][3];
		solve();
	}

	// helper methods for chip localisation and transformation
	/**
	 * Display contour defined by given contour
	 * 
	 * @param outer given contour
	 * @return image where areas outside contour are black, inside are white
	 */
	private Mat displayContour(MatOfPoint contour) {
		Mat panel = Mat.zeros(gray.size(), 0);
		ArrayList<MatOfPoint> ct = new ArrayList<MatOfPoint>();
		ct.add(contour);
		Imgproc.drawContours(panel, ct, 0, new Scalar(255), -1);
		return panel;
	}

	/**
	 * Returns all contours detected in "target" image in ascending area order
	 * 
	 * @param target target gray scale image for contour detection
	 * @return a list of contour objects with ascending area
	 */
	private List<MatOfPoint> contourFinder(Mat target, int threshold) {
		Mat thresh = new Mat(), inv = new Mat(), hierarchy = new Mat();
		Imgproc.threshold(target, thresh, threshold, 255, 0); // thresholding
		Core.bitwise_not(thresh, inv); // find inverse
		List<MatOfPoint> contour = new ArrayList<MatOfPoint>();

		Imgproc.findContours(inv, contour, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
		Collections.sort(contour, (c1, c2) -> (int) (Imgproc.contourArea(c1)) - (int) (Imgproc.contourArea(c2)));
		return contour;
	}

	/**
	 * Convert MatOfPoint2f to list of MatOfPoint
	 * 
	 * @param corners corners in MatOfPoint2f format
	 * @return a list of corners each of MatOfPoint format
	 */
	private List<MatOfPoint> mfToLmop(MatOfPoint2f corners) {
		List<MatOfPoint> res = new ArrayList<MatOfPoint>();
		MatOfPoint hold = new MatOfPoint();
		hold.fromList(corners.toList());
		res.add(hold);
		return res;
	}

	/**
	 * Find corners of given contour "tar" and draw
	 * 
	 * @param tar  target contour for corner detection
	 * @param thre approximation threshold. .05-rough / .01-fine
	 * @return corners detected
	 * @throws IOException
	 */
	private MatOfPoint2f cornerFinder(MatOfPoint tar, double thre) throws IOException {
		MatOfPoint2f ct = new MatOfPoint2f(tar.toArray()); // target contour to draw
		double epsilon = thre * Imgproc.arcLength(ct, true); //
		MatOfPoint2f corners = new MatOfPoint2f();
		Imgproc.approxPolyDP(ct, corners, epsilon, true); // derive corners of ct
		return corners;
	}

	/**
	 * Self-explanatory
	 * 
	 * @param mask/img given Mat to be inversely merged
	 * @return merge image where areas inside mask are maintained, outside are black
	 */
	private Mat maskInvAdd(Mat mask, Mat img) {
		Mat mask_inv = new Mat(), img_inv = new Mat(), merge = new Mat(), merge_inv = new Mat();
		Core.bitwise_not(mask, mask_inv);
		Core.bitwise_not(img, img_inv);
		Core.add(mask_inv, img_inv, merge_inv);
		Core.bitwise_not(merge_inv, merge);
		return merge;
	}

	/**
	 * Order points in clockwise order
	 * 
	 * @param mixed corners in random order
	 * @return corners in clockwise order
	 */
	private MatOfPoint2f orderPoint(MatOfPoint2f mixed) {
		List<Point> cls = mixed.toList(); // list of corner points
		assert (cls.size() == 4);
		Collections.sort(cls, (c1, c2) -> (int) (c1.x - c2.x)); // sort corners based on x-coordinates
		Point p2 = cls.get(2), p3 = cls.get(3); // p3-rightmost, p2-second

		Point tl, bl, tr = p3, br = p2;
		tl = cls.get(0).y - cls.get(1).y < 0 ? cls.get(0) : cls.get(1); // top-left corner, small y
		bl = cls.get(0).y - cls.get(1).y > 0 ? cls.get(0) : cls.get(1); // bottom-left corner, big y
		assert (p2.x != tl.x && p3.x != tl.x);
		double s2 = (tl.y - p2.y) / (tl.x - p2.x);
		double s3 = (tl.y - p3.y) / (tl.x - p3.x);
		tr = s2 < s3 ? p2 : p3;
		br = s2 < s3 ? p3 : p2;

		MatOfPoint2f res = new MatOfPoint2f(tl, tr, br, bl);
		return res;
	}

	/**
	 * Obtain areas being white in a but black in b
	 * 
	 * @param a
	 * @param b
	 * @return
	 */
	private Mat subtract(Mat a, Mat b) {
		Mat a_inv = new Mat(), res_inv = new Mat(), res = new Mat();
		Core.bitwise_not(a, a_inv);
		Core.add(a_inv, b, res_inv);
		Core.bitwise_not(res_inv, res);
		// show(res, .1);
		return res;
	}

	/**
	 * Draw polygon using corners of mj/mr_Corners
	 * 
	 * @param id = 0, use mjC; otherwise use mrC;
	 * @return polygon defined by mj/mr_Corners
	 */
	private Mat drawPolygon(int id) {
		Mat polygon = Mat.zeros(gray.size(), 0);
		if (id == 0)
			Imgproc.polylines(polygon, mfToLmop(mjCorners), true, new Scalar(255), 3);
		else
			Imgproc.polylines(polygon, mfToLmop(mrCorners), true, new Scalar(255), 3);
		// show(mjPolygon, .1); // polygon defined by mjCorners
		return polygon;
	}

	/**
	 * Check whether given contour is the outer contour of the chip - update inner
	 * 
	 * @param outer given contour
	 * @return true if outer covers chip and update res; false otherwise
	 * @throws IOException
	 */
	private boolean check(MatOfPoint outer, int threshold) throws IOException {
		MatOfPoint2f corners = cornerFinder(outer, .05);
		if (corners.toList().size() != 4)
			return false;

		Mat panel = displayContour(outer);
		Mat zoomIn = maskInvAdd(panel, gray);
		List<MatOfPoint> cts = contourFinder(zoomIn, threshold);
		MatOfPoint postu_inner = cts.get(cts.size() - 2);
		corners = cornerFinder(outer, .05);
		if (corners.toList().size() != 4)
			return false;

		inner = postu_inner;
		return true;
	}

	/**
	 * Add scl to every pixel of img
	 * 
	 * @param img, input image
	 * @param scl, scalar to be added
	 * @return img added with scl
	 */
	private Mat addToChannel(Mat img, Scalar scl) {
		Size sz = img.size();
		Mat addition = new Mat((int) sz.height, (int) sz.width, CvType.CV_8UC3, scl);
		Mat res = new Mat();
		Core.add(img, addition, res);
		return res;
	}

	// chip localisation and transformation
	/**
	 * Find corners of major and minor chip area - update mjCorners, mrCorners
	 * 
	 * @throws IOException
	 */
	public void corner(int threshold) throws IOException {
		mjCorners = orderPoint(cornerFinder(inner, .05)); // 4/5 corners of chip
		Mat mjPolygon = drawPolygon(0);
		List<MatOfPoint> mjCt = contourFinder(mjPolygon, threshold);
		Mat mjChip = displayContour(mjCt.get(0));

		// find minor chip corners
		Mat chip = displayContour(inner); // binary image where areas outside inner are white
		Mat mrChip = subtract(chip, mjChip); // chip subtract mjChip
		Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(10, 10));
		Imgproc.erode(mrChip, mrChip, element); // erosion to avoid linkage between mr and mj
		List<MatOfPoint> mrCt = contourFinder(mrChip, threshold);
		mrChip = displayContour(mrCt.get(mrCt.size() - 2));
		mrCorners = cornerFinder(mrCt.get(mrCt.size() - 2), .01);
	}

	/**
	 * Perspectively transform chip - update restr
	 * 
	 * @throws IOException
	 */
	public void chipTransform() throws IOException {
		PersTransformation pt = new PersTransformation(orig, 420, mjCorners, mrCorners);
		pt.cornerTag();
		Mat warp = pt.transform();

		// black-out the top right triangle
		Mat a = new Mat(), thresh = new Mat(), restr_inv = new Mat();
		Core.bitwise_not(warp, a);
		Imgproc.threshold(a, thresh, 200, 255, 0);
		Core.add(thresh, a, restr_inv);
		Core.bitwise_not(restr_inv, restr);
	}

	/**
	 * Recursively check whether chip can be located given current threshold value
	 * (and L channel)
	 * 
	 * @param threshold, current thresholding value for contour detection
	 * @return whether chip can be located - update inner
	 * @throws IOException
	 */
	public int recur(int threshold) throws IOException {
		List<MatOfPoint> sortedContour = contourFinder(gray, threshold);
		int n = sortedContour.size(), i = n - 1, ct = 0;
		while (ct <= 3) {
			ct++;
			MatOfPoint outer = sortedContour.get(i);
			boolean reach = check(outer, threshold); // update inner
			if (reach)
				return 1;
		}
		return 0;
	}

	/**
	 * Read in image and detect chip area - update orig, gray, inner
	 * 
	 * @param filename  filename for the image to be read
	 * @param threshold thresholding value
	 * @return
	 * @throws IOException
	 */
	public int readin() throws IOException {
		this.orig = Imgcodecs.imread(filename);
		this.gray = new Mat();
		Imgproc.cvtColor(orig, this.gray, Imgproc.COLOR_BGR2GRAY);

		Mat lab = new Mat();
		Imgproc.cvtColor(orig, lab, Imgproc.COLOR_BGR2Lab);
		Scalar gms = Core.mean(gray);
		int graymean = (int) gms.val[0], ct = 0;
		// increment L channel by 10 at each iteration
		while (ct++ <= 10) {
			lab = addToChannel(lab, new Scalar(0, 0, 10));
			Mat cp = new Mat();
			Imgproc.cvtColor(lab, cp, Imgproc.COLOR_Lab2BGR);
			int thre = 90;
			// increment threshold value by 10 at each iteration
			while (thre <= graymean) {
				thre += 10;
				int isChip = recur(thre);
				// pass two contour check
				if (isChip == 0)
					continue;
				// transform
				try {
					corner(thre);
					chipTransform();
				} catch (Exception e) {
					continue;
				}
				return 1;
			}
		}
		return 0;
	}

	/**
	 * Remediate restr to black-out residual black borders and flash glares.
	 * Necessary to perform, but can be inaccurate, improvements needed.
	 * 
	 * @throws IOException
	 */
	public void remediate() throws IOException {
		// prepare L channel image
		Mat lab = new Mat(), lc = new Mat(), lc_inv = new Mat();
		Imgproc.cvtColor(restr, lab, Imgproc.COLOR_BGR2Lab);
		Core.extractChannel(lab, lc, 0); // get L channel
		Core.bitwise_not(lc, lc_inv);

		// evaluate local brightness, dividing L channel chip evenly into four blocks
		double meanll = 256, meanlh = 0;
		for (int rlo = 0; rlo <= size[0] / 2; rlo += size[0] / 2 + 1) { // low bound for row number
			for (int clo = 0; clo <= size[1] / 2; clo += size[1] / 2 + 1) { // ... column ...
				int rhi = rlo + size[0] / 2, chi = clo + size[1] / 2; // high bound ...
				Mat cur = lc.submat(rlo, rhi, clo, chi);
				MinMaxLocResult mmlr = Core.minMaxLoc(cur);
				meanll = Math.min(meanll, mmlr.minVal);
				meanlh = Math.max(meanlh, mmlr.maxVal);
			}
		}

		// derive threshold value
		double thb = 80, thg = 205; // empirically decided border/glare threshold
		if (meanll >= 80) // if the image is too dark, can hardly discern border
			thb = meanll - (meanll - 80) * .6;
		if (meanlh < 205) // if image too bright, can hardly discern glare
			thg = thg - (205 - meanlh);

		// thresholding
		Mat borderMask = new Mat(), glareMask = new Mat();
		Mat mask = new Mat(lc.size(), CvType.CV_8UC1, new Scalar(255));
		// border mask
		Imgproc.threshold(lc, borderMask, thb, 1, 0); // pixels originally >/<= thb are converted to 1/0
		ImageKit ik = new ImageKit(borderMask, 1);
		Core.multiply(borderMask, mask, mask);
		// glare mask
		Imgproc.threshold(lc_inv, glareMask, 255 - thg, 1, 0); // pixels originally </>= thb are converted to 1/0
		Core.multiply(glareMask, mask, mask);
		ik.show(mask);
		// apply mask
		Imgproc.cvtColor(mask, mask, Imgproc.COLOR_GRAY2BGR);
		remediated = maskInvAdd(restr, mask);
		// ik.show(remediated);
		// System.out.println(thg + " " + thb + " " + meanll + " " + meanlh);
	}

	// block segmentation
	/**
	 * Segment restr into rw * cl blocks - update blocks, assem
	 * 
	 * @throws IOException
	 */
	public void blockize() throws IOException {
		for (int rid = 0; rid < size[0]; rid++) {
			for (int cid = 0; cid < size[1]; cid++) {
				int id = rid * size[1] + cid;
				int x = size[2] + cid * size[4] + (2 * cid + 1) * size[8];
				int y = size[3] + rid * size[5] + (2 * rid + 1) * size[8];
				int a = rid == size[0] - 1 ? size[9] : y + size[7]; // special treatment for last row
				
				// segment and reassemble - update blocks & assem
				Mat block = restr.submat(y - size[7], a, x - size[6], x + size[6]); // top, bottom, left, right boundary
				blocks[id] = block;
				blocks[id].copyTo(assem.submat(y - size[7], a, x - size[6], x + size[6])); // assemble blocks back to intact chip
				
				// derive block infomation
				ImageKit ik = new ImageKit(block, 1, 7);
				normBlocks[id] = ik.norm; // normalised block
				blockValue[id] = ik.repreValue; // representative value of spot in current block
			}
		}
	}

	// All in one
	public void solve() throws IOException {
		int success = readin();
		if (success == 0)
			throw new IllegalArgumentException(
					"Image cannot be transformed, please ensure the border is not interrupted by glare");
		assem = new Mat(restr.size(), CvType.CV_8UC3, new Scalar(0, 0, 0));
		blockize();
		remediate();
		// Imgcodecs.imwrite("spot.jpg", a);
	}

	public static void main(String[] args) throws IOException {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		String[] nms = { "D:/Software/Python37/wd/2018-19FYP/chip_v2/Benchmarking/20190305/version2/1-orig/light3/t1l3_20190305_18_lighttray_900lux_1min_TET0.5.jpg",
				         "D:/Software/Python37/wd/2018-19FYP/chip_v2/Benchmarking/20190305/version2/1-orig/light1/t1l1_20190305_07_bench_330lux_4min_AMP2.jpg",
				         "D:/Software/Python37/wd/2018-19FYP/chip_v2/Benchmarking/20190305/version2/1-orig/light1/t1l1_20190305_35_bench_330lux_3min_LB.jpg"
				       };
		int[] size = { 7, 6, 20, 27, 40, 30, 35, 30, 15, 420 };
		
		// TEST
		Chip chip = new Chip(nms[2], size);
		System.out.println(chip.mjCorners.toList().toString());
		System.out.println(chip.mrCorners.toList().toString());
		
		ImageKit imgdis = new ImageKit(chip.assem, 1);
		imgdis.show(chip.assem);
		imgdis.show(chip.restr);
		// block info check
		Mat b5 = chip.blocks[5], b5r = new Mat();
		imgdis.show(chip.blocks[5]);
		Core.extractChannel(b5, b5r, 0); imgdis.show(b5r);
	}
}