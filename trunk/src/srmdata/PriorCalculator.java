package srmdata;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.TermDocs;
import org.apache.lucene.index.TermEnum;
import org.apache.lucene.index.TermFreqVector;

public class PriorCalculator {

	static final int numThreads = 4;

	int nTrainDocs;
	int nTestDocs;
	int collectionSize;
	IndexReader trainIR;
	IndexReader testIR;
	String fieldName;
	List<Term> allTerms;
	Map<Integer,Integer> doc_length;
	double[][][] modelScores;

	public PriorCalculator(IndexReader testIR, IndexReader trainIR, String fieldName) {
		this.trainIR = trainIR;
		this.testIR = testIR;
		this.fieldName = fieldName;
	}
	
	double[][] computePriors() throws Exception {

		// assume there are no holes in document ids for train/test indices
		nTrainDocs = trainIR.numDocs();
		nTestDocs = testIR.numDocs();

		// find number of terms in all training documents for the given field
		doc_length = new HashMap<Integer, Integer>();
		for (int docID = 0; docID < nTrainDocs; ++docID) {
			if (trainIR.document(docID) == null)
				continue;
			doc_length.put(docID, getNumTermsInDocument(trainIR,docID,fieldName));
		}

		modelScores = new double[numThreads][nTrainDocs][nTestDocs];
		for (int threadNum = 0; threadNum < numThreads; ++threadNum) {
			for (int i = 0; i < modelScores[threadNum].length; ++i)
				for (int j = 0; j < modelScores[threadNum][i].length; ++j)
					modelScores[threadNum][i][j] = 1.0;
		}

		collectionSize = findCollectionSize(trainIR, fieldName);
		allTerms = new ArrayList<Term>();

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
			allTerms.add(t);
		}
		terms.close();

		int fromIndex = 0;
		int toIndex = -1;

		int numTerms = allTerms.size();
		int chunkSize = (int) Math.ceil((double)numTerms / numThreads);

		System.out.println("Number of Terms: " + numTerms);
		System.out.println("Chunk Size: " + chunkSize);
		
		List<Thread> threads = new ArrayList<Thread>();
		for (int threadNum = 0; threadNum < numThreads; ++threadNum) {
			fromIndex = threadNum*chunkSize;
			toIndex = fromIndex + chunkSize - 1;
			if (fromIndex < numTerms) {
				toIndex = (toIndex < numTerms) ? toIndex : numTerms-1;
				System.out.println("Worker " + threadNum + " fromIndex: " + fromIndex + " toIndex: " + toIndex);
				Runnable runnable = new PriorCalculatorRunnable(fromIndex, toIndex, modelScores[threadNum]);
				Thread worker = new Thread(runnable);
				worker.start();
				threads.add(worker);
			}
		}

		for (Thread thread : threads) {
			thread.join();
		}

		double[][] finalModelScores = new double[nTrainDocs][nTestDocs];
		for (int i = 0; i < finalModelScores.length; ++i) {
			for (int j = 0; j < finalModelScores[i].length; ++j) {
				finalModelScores[i][j] = 1.0;
			}
		}
		for (int threadNum = 0; threadNum < numThreads; ++threadNum) {
			for (int i = 0; i < finalModelScores.length; ++i) {
				for (int j = 0; j < finalModelScores[i].length; ++j) {
					finalModelScores[i][j] *= modelScores[threadNum][i][j];
					if (modelScores[threadNum][i][j] == 0.0) {
						System.out.println("here " + threadNum + " " + i + " " + j);
						System.exit(0);
					}
				}
			}
		}
		
		return finalModelScores;
	}

	int getNumTermsInDocument(IndexReader ir, int docID, String fieldName) throws IOException {
		TermFreqVector tfVec = ir.getTermFreqVector(docID, fieldName);
		return ((tfVec == null) ? 0 : tfVec.size());
	}

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

	boolean containsNumber(String str) {
		for (int i = 0; i < str.length(); ++i) {
			if (str.charAt(i) >= '0' && str.charAt(i) <= '9')
				return true;
		}
		return false;
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
			if (mlEstimates[md] == 0.0)
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
		return 
			Arrays.asList(pavg, meanfreq, collectionFreq);
	}

	class PriorCalculatorRunnable implements Runnable {

		int fromIndex;
		int toIndex;
		double[][] localModelScores;

		public PriorCalculatorRunnable(int fromIndex, int toIndex, double[][] localModelScores) {
			this.fromIndex = fromIndex;
			this.toIndex = toIndex;
			this.localModelScores = localModelScores;
			for (int i = 0; i < localModelScores.length; ++i)
				for (int j = 0; j < localModelScores[i].length; ++j)
					localModelScores[i][j] = 1.0;
		}

		@Override
		public void run() {
			try {
				double score[] = new double[2];
				for (int index = fromIndex; index <= toIndex; ++index) {
					Term t = allTerms.get(index);
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
	
	//				long t1 = System.nanoTime();
					for (int md = 0; md < nTrainDocs; ++md) {
						score[0] = mle[md];
						score[1] = 1.0 - score[0];
						for (int q = 0; q < nTestDocs; ++q)
							localModelScores[md][q] *= score[termDocsArr[q]];
					}
	//				long t2 = System.nanoTime();
	//				System.out.println("Time Taken: " + (t2-t1)/1E6);
				}
			}
			catch (Exception E) {
				System.err.println("Exception Caught: " + E.getMessage());
				System.exit(0);
			}
		}
	}
}
