package srmdata;

import java.util.ArrayList;
import java.util.List;

import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.TermDocs;
import org.apache.lucene.index.TermEnum;

public class PriorCalculator {

	static final int numThreads = 4;

	int nTrainDocs;
	int nTestDocs;
	int collectionSize;
	IndexReader trainIR;
	IndexReader testIR;
	String fieldName;
	List<Term> allTerms;
	int[] doc_lengths;
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
		doc_lengths = new int[nTrainDocs];

		modelScores = new double[numThreads][nTrainDocs][nTestDocs];
		for (int threadNum = 0; threadNum < numThreads; ++threadNum) {
			for (int i = 0; i < modelScores[threadNum].length; ++i)
				for (int j = 0; j < modelScores[threadNum][i].length; ++j)
					modelScores[threadNum][i][j] = 1.0;
		}

		collectionSize = StructuredRelevanceModel.findCollectionSize(trainIR, fieldName, doc_lengths);
		allTerms = new ArrayList<Term>();

		TermEnum terms = trainIR.terms();
//		boolean trainDone = false;
		while (true) {
			boolean hasNext = terms.next();
			if (!hasNext) {
				terms.close();
//				if (trainDone)
					break;
//				terms = trainIR.terms();
//				trainDone = true;
//				continue;
			}
			Term t = terms.term();
			if (!t.field().equals(fieldName) || StructuredRelevanceModel.containsNumber(t.text()))
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
//					if (modelScores[threadNum][i][j] == 0.0) {
//						System.out.println("here " + threadNum + " " + i + " " + j);
//						System.exit(0);
//					}
				}
			}
		}
		
		return finalModelScores;
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
				double[] mle = new double[nTrainDocs];
				for (int index = fromIndex; index <= toIndex; ++index) {
					Term t = allTerms.get(index);
					StructuredRelevanceModel.compute_mlestimate(trainIR, fieldName, t, doc_lengths, collectionSize, mle);
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
