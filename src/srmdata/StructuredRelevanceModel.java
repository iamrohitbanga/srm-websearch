package srmdata;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.TermDocs;
import org.apache.lucene.index.TermEnum;
import org.apache.lucene.index.TermFreqVector;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.RAMDirectory;

public class StructuredRelevanceModel {

	Set<Integer> allDocIds;
	LinkedHashSet<Integer> testDocIds;
	Set<Integer> trainDocIds;
	
	public StructuredRelevanceModel() {
		allDocIds = new LinkedHashSet<Integer>();
		testDocIds = new LinkedHashSet<Integer>();
		trainDocIds = new LinkedHashSet<Integer>();
	}


	class Score {
		double score;
		int docID;
	}

	void predictField(String fieldToPredict, Map<String,String> testTrainFiles) throws Exception {

		for (Map.Entry<String, String> filenames : testTrainFiles.entrySet()) {

			RAMDirectory trainRAMDirectory = new RAMDirectory(FSDirectory.open(new File(filenames.getKey()))); 
			RAMDirectory testRAMDirectory = new RAMDirectory(FSDirectory.open(new File(filenames.getValue()))); 
			IndexReader trainIR = IndexReader.open(trainRAMDirectory, true);
			IndexReader testIR  = IndexReader.open(testRAMDirectory, true);

			System.out.println("Train File Name: " + filenames.getKey() + " Test File Name: " + filenames.getValue());
			int num_fields = 3;

			int nTrainDocs = trainIR.numDocs();
			int nTestDocs = testIR.numDocs();

			double[][][] scores = new double[num_fields][][];
			long t1, t2;
			t1 = System.nanoTime();
				scores[0] = computePriors(testIR, trainIR, "title");
			t2 = System.nanoTime();
			System.out.println("Time Taken Priors (title): " + ((double)(t2-t1)) / 1E9);
			t1 = System.nanoTime();
				scores[1] = computePriors(testIR, trainIR, "desc");
			t2 = System.nanoTime();
			System.out.println("Time Taken Priors (desc): " + ((double)(t2-t1)) / 1E9);
			t1 = System.nanoTime();
//			scores[2] = computePriors(testIR, trainIR, "content");
//			t2 = System.nanoTime();
//			System.out.println("Time Taken Priors (content): " + ((double)(t2-t1)) / 1E9);

			Score[][] combined_score = new Score[nTestDocs][nTrainDocs];
			for (int i = 0; i < nTestDocs; ++i) {
				for (int j = 0; j < nTrainDocs; ++j) {
					combined_score[i][j] = new Score();
				}
			}

			for (int i = 0; i < nTrainDocs; ++i) {
				for (int j = 0; j < nTestDocs; ++j) {
					combined_score[j][i].docID = j;
					combined_score[j][i].score = scores[0][i][j] * scores[1][i][j];
				}
			}

			t1 = System.nanoTime();
			DescendingComp comp = new DescendingComp();
			for (int i = 0; i < nTestDocs; ++i) {

				double total_score = 0.0;
				for (int j = 0; j < nTrainDocs; ++j)
					total_score += combined_score[i][j].score;

				for (int j = 0; j < nTrainDocs; ++j)
					combined_score[i][j].score /= total_score;

				Arrays.sort(combined_score[i], comp);
			}
			t2 = System.nanoTime();
			System.out.println("Time Taken Normalization and Sorting: " + ((double)(t2-t1)) / 1E9);

			trainIR.close();
			testIR.close();
		}
	}

	class DescendingComp implements Comparator<Score> {
		@Override
		public int compare(Score o1, Score o2) {
			Double diff = o2.score-o1.score;
			if (diff < 0)
				return -1;
			if (diff > 0)
				return 1;
			return 0;
		}
	}
	
	boolean containsNumber(String str) {
		for (int i = 0; i < str.length(); ++i) {
			if (str.charAt(i) >= '0' && str.charAt(i) <= '9')
				return true;
		}
		return false;
	}

	int getNumTermsInDocument(IndexReader ir, int docID, String fieldName) throws IOException {
		TermFreqVector tfVec = ir.getTermFreqVector(docID, fieldName);
		return ((tfVec == null) ? 0 : tfVec.size());
	}

	double[][] computePriors(IndexReader testIR, IndexReader trainIR, String fieldName) throws Exception {

		// assume there are no holes in document ids for train/test indices
		int nTrainDocs = trainIR.numDocs();
		int nTestDocs = testIR.numDocs();

		// find number of terms in all training documents for the given field
		Map<Integer,Integer> doc_length = new HashMap<Integer, Integer>();
		for (int docID = 0; docID < nTrainDocs; ++docID) {
			if (trainIR.document(docID) == null)
				continue;
			doc_length.put(docID, getNumTermsInDocument(trainIR,docID,fieldName));
		}

		double[][] modelScores;
		modelScores = new double[nTrainDocs][nTestDocs];
		for (int i = 0; i < modelScores.length; ++i)
			for (int j = 0; j < modelScores[i].length; ++j)
				modelScores[i][j] = 1.0;

		int collectionSize = findCollectionSize(trainIR, fieldName);

		double score[] = new double[2];
		TermEnum terms = testIR.terms();
		boolean trainDone = false;
		while (true) {

			boolean hasNext = terms.next();
			if (!hasNext) {
				terms.close();
				if (trainDone)
					break;
				terms = trainIR.terms();
				trainDone = true;
				continue;
			}

			Term t = terms.term();
			if (!t.field().equals(fieldName) || containsNumber(t.text()))
				continue;

			double[] mle = compute_mlestimate(trainIR, fieldName, t, doc_length, collectionSize);
			if (mle == null)
				continue;

			int[] termDocsArr = new int[nTestDocs];
			for (int i = 0; i < termDocsArr.length; ++i)
				termDocsArr[i] = 1;
			TermDocs termDocs = testIR.termDocs(t);
			while (termDocs.next()) {
				termDocsArr[termDocs.doc()] = 0;
			}
			termDocs.close();

//			long t1 = System.nanoTime();
			for (int md = 0; md < nTrainDocs; ++md) {
				score[0] = mle[md];
				score[1] = 1.0 - score[0];
				for (int q = 0; q < nTestDocs; ++q)
					modelScores[md][q] *= score[termDocsArr[q]];
			}
//			long t2 = System.nanoTime();
//			System.out.println("Time Taken: " + (t2-t1)/1E6);
		}

		terms.close();
		return modelScores;
	}

	private double[] compute_mlestimate(IndexReader ir, String fieldName,
			Term t, Map<Integer, Integer> doc_length, int collectionSize) throws Exception {

		List<Double> avgs = compute_avgs(ir, fieldName, t);
		Double pavg = avgs.get(0);
		Double meanfreq = avgs.get(1);
		Double collectionFreq = avgs.get(2);

		if (collectionFreq == 0.0) {
			return null;
		}
		
//		Map<Integer,Double> mlEstimates = new HashMap<Integer, Double>();
		double[] mlEstimates = new double[ir.maxDoc()];

		double term1 = meanfreq / (1.0 + meanfreq);
		double term2 = 1.0 / (1.0 + meanfreq);
		TermDocs termDocs = ir.termDocs(t);
		while (termDocs.next()) {
			int d = termDocs.doc();
			int tf = termDocs.freq();
			double R = term2 * Math.pow(term1,tf);
			double pml = ((double)tf) / doc_length.get(d);
			double val = Math.pow(pml, 1.0-R) * Math.pow(pavg, R);
			mlEstimates[d] = val;
		}
		termDocs.close();

		double defaultVal = ((double)collectionFreq) / collectionSize;
		for (int md = 0; md < ir.maxDoc(); ++md) {
			if (mlEstimates[md] != 0.0)
				mlEstimates[md] = defaultVal;
		}

		return mlEstimates;
	}

	private List<Double> compute_avgs(IndexReader ir, String fieldName,
			Term t) throws Exception {

		double collectionFreq = 0;
		double pavg = 0.0;
		double meanfreq = 0.0;

		int count = 0;
		TermDocs termDocs = ir.termDocs(t);
		while (termDocs.next()) {

			int d = termDocs.doc();
			int tf = termDocs.freq();
			TermFreqVector tfv = ir.getTermFreqVector(d, fieldName);
			int dl = tfv.size();
			if (dl == 0)
			{
				System.out.println("d:" + d + " dl:" + dl);
				System.exit(0);
			}
			double pml = ((double)tf) / dl;
			pavg = pavg + pml;
			meanfreq = meanfreq + tf;
			collectionFreq = collectionFreq + tf;
			count++;
		}
		termDocs.close();

		if (count == 0) {
			pavg = 0.0;
			meanfreq = 0.0;
		}
		else {
			pavg = pavg / count;
			meanfreq = meanfreq / count;
		}
		return Arrays.asList(pavg, meanfreq, collectionFreq);
	}

	/**
	 * Find total number of tokens in the collection
	 * @param ir
 	 * @param fieldName
	 * @throws IOException
	 */
	private Integer findCollectionSize(IndexReader ir, String fieldName) throws IOException {
		int collectionSize = 0;
		TermEnum terms = ir.terms();
		while (terms.next()) {
			Term t = terms.term();
			if (!t.field().equals(fieldName) || containsNumber(t.text()))
				continue;
			TermDocs termDocs = ir.termDocs(t);
			while (termDocs.next()) {
				collectionSize += termDocs.freq();
			}
			termDocs.close();
		}
		terms.close();
		return collectionSize;
	}
}
