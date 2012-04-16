package srmdata;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.Fieldable;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.NumericRangeQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.RAMDirectory;

import srmdata.StructuredRelevanceModel.DescendingScoreComp;
import srmdata.StructuredRelevanceModel.Score;

public class MultiLabelClassificationNSDL {

	private static final String TEST_INDEX_NAME = "../../multi_label_test_index";
	private static final String TRAIN_INDEX_NAME = "../../multi_label_train_index";
	private static int numTesting = 100;
	private static int numTraining = 5000;

	public static void main(String[] args) throws Exception {

		long t1, t2;

		t1 = System.nanoTime();		
			generateMultiLabelDataset();
		t2 = System.nanoTime();
		System.out.println("Time Taken for Generation: " + (t2-t1)/1E9);

		t1 = System.nanoTime();
			predictfield("subject");
		t2 = System.nanoTime();
		System.out.println("Time Taken for Subject Prediction: " + (t2-t1)/1E9);
	}

	private static void predictfield(String fieldToPredict) throws Exception {

		RAMDirectory trainRAMDirectory = new RAMDirectory(FSDirectory.open(new File(TRAIN_INDEX_NAME))); 
		RAMDirectory testRAMDirectory = new RAMDirectory(FSDirectory.open(new File(TEST_INDEX_NAME))); 
		IndexReader trainIR = IndexReader.open(trainRAMDirectory, true);
		IndexReader testIR  = IndexReader.open(testRAMDirectory, true);

		int nTrainDocs = trainIR.numDocs();
		int nTestDocs = testIR.numDocs();

		long t1, t2;

		int num_fields = 3;
		double[][][] scores = new double[num_fields][][];
		StructuredRelevanceModel srm = new StructuredRelevanceModel();
		t1 = System.nanoTime();
		scores[0] = srm.computePriors(testIR, trainIR, "title");
//		PriorCalculator priorCalcTitle = new PriorCalculator(testIR, trainIR, "title");
//		scores[0] = priorCalcTitle.computePriors();
		t2 = System.nanoTime();
		System.out.println("Time Taken Priors (title): " + ((double)(t2-t1)) / 1E9);
		t1 = System.nanoTime();
			scores[1] = srm.computePriors(testIR, trainIR, "desc");
//		PriorCalculator priorCalcDesc = new PriorCalculator(testIR, trainIR, "desc");
//		scores[1] = priorCalcDesc.computePriors();
		t2 = System.nanoTime();
		System.out.println("Time Taken Priors (desc): " + ((double)(t2-t1)) / 1E9);
		t1 = System.nanoTime();
//			scores[2] = srm.computePriors(testIR, trainIR, "content");
//		PriorCalculator priorCalcContent = new PriorCalculator(testIR, trainIR, "content");
//		scores[2] = priorCalcContent.computePriors();
		t2 = System.nanoTime();
		System.out.println("Time Taken Priors (content): " + ((double)(t2-t1)) / 1E9);

		Score[][] combined_score = new Score[nTestDocs][nTrainDocs];
		for (int i = 0; i < nTestDocs; ++i) {
			for (int j = 0; j < nTrainDocs; ++j) {
				combined_score[i][j] = new Score();
			}
		}

		for (int i = 0; i < nTrainDocs; ++i) {
			for (int j = 0; j < nTestDocs; ++j) {
				combined_score[j][i].docID = i;
				combined_score[j][i].score = scores[0][i][j] * scores[1][i][j];// * scores[2][i][j];
//				if (scores[1][i][j] == 0.0) {
//					System.out.print("Score is Zero: " + i + " " + j + " ");
//				}
			}
		}

		t1 = System.nanoTime();
		DescendingScoreComp comp = new DescendingScoreComp();
		for (int i = 0; i < nTestDocs; ++i) {
			Arrays.sort(combined_score[i], comp);
		}

		int topN = 100;
//		for (int i = 0; i < nTestDocs; ++i) {
//			double total_score = 0.0;
//			for (int j = 0; j < nTrainDocs; ++j) {
//				total_score += combined_score[i][j].score;
//			}
//
//			if (total_score == 0.0) {
//				for (int j = 0; j < topN; ++j)
//					combined_score[i][j].score = 0.0;
//			}
//			else {
//				for (int j = 0; j < topN; ++j)
//					combined_score[i][j].score /= total_score;
//			}
//		}
		t2 = System.nanoTime();
		System.out.println("Time Taken Normalization and Sorting: " + ((double)(t2-t1)) / 1E9);

		Map<String,Double> fieldCounts = new HashMap<String,Double>();
		for (int i = 0; i < nTestDocs; ++i) {
			fieldCounts.clear();
			for (int j = 0; j < topN; ++j) {
				int docID = combined_score[i][j].docID;
				combined_score[i][j].relevance = 0.0;
				Document doc = trainIR.document(docID);
				Fieldable[] fields = doc.getFieldables(fieldToPredict);
				for (Fieldable field : fields) {
					String fieldValue = field.stringValue();
					Double count = fieldCounts.get(fieldValue);
					if (count == null)
						count = 1.0 * combined_score[i][j].score;
					else
						count = count + combined_score[i][j].score;
					fieldCounts.put(fieldValue, count);
				}
			}

			List<Relevance> relevanceModel = new ArrayList<Relevance>();
			for (String fieldValue : fieldCounts.keySet()) {
				Relevance relevance = new Relevance();
				relevance.fieldValue = fieldValue;
				relevance.score = fieldCounts.get(fieldValue);
				relevanceModel.add(relevance);
			}
			Collections.sort(relevanceModel, new DescendingRelevanceComp());

			Document testDoc = testIR.document(i);

			System.out.print("docID:" + i + ":");
			for (Fieldable fieldable : testDoc.getFieldables(fieldToPredict)) {
				System.out.print(fieldable.stringValue() + ":");
			}
			System.out.println(":");
			
//			int max = (relevanceModel.size() < 10) ? relevanceModel.size() : 10;
			int max = 6;
			for (int j = 0; j < max; ++j) {
				Relevance relevance = relevanceModel.get(j);
				System.out.println("    " + relevance.fieldValue + "(" + relevance.score + ");");
			}
			System.out.println();
		}

		trainIR.close();
		testIR.close();
	}

	static class Relevance {
		public String fieldValue;
		public double score = 0.0;
	}

	static class DescendingRelevanceComp implements Comparator<Relevance> {
		@Override
		public int compare(Relevance o1, Relevance o2) {
			Double diff = o2.score-o1.score;
			if (diff < 0)
				return -1;
			if (diff > 0)
				return 1;
			return 0;
		}
	}
	

	private static void generateMultiLabelDataset() throws Exception {

		File nsdl_global_file = new File(NSDLIndex.NSDL_GLOBAL_INDEX_DIR_NAME);
		IndexReader ir = IndexReader.open(FSDirectory.open(nsdl_global_file), true);

		IndexWriter testIW = NSDLIndex.createIndexWriter(TEST_INDEX_NAME);
		IndexWriter trainIW = NSDLIndex.createIndexWriter(TRAIN_INDEX_NAME);
		testIW.deleteAll();
		trainIW.deleteAll();

		// first search for all documents having at least one value for each of
		// content, description, title, subject, audience fields
		IndexSearcher searcher = new IndexSearcher(ir);

		NumericRangeQuery<Integer> nq1 = NumericRangeQuery.newIntRange("num_subject", 1, 100000, true, true);
		NumericRangeQuery<Integer> nq3 = NumericRangeQuery.newIntRange("num_audience", 1, 10, true, true);
		NumericRangeQuery<Integer> nq4 = NumericRangeQuery.newIntRange("title_len", 1, 1000000, true, true);
		NumericRangeQuery<Integer> nq5 = NumericRangeQuery.newIntRange("content_len", 1, 100000000, true, true);
		NumericRangeQuery<Integer> nq6 = NumericRangeQuery.newIntRange("desc_len", 1, 100000000, true, true);

		BooleanQuery nq = new BooleanQuery();
		nq.add(nq1, BooleanClause.Occur.MUST);
		nq.add(nq3, BooleanClause.Occur.MUST);
		nq.add(nq4, BooleanClause.Occur.MUST);
		nq.add(nq5, BooleanClause.Occur.MUST);
		nq.add(nq6, BooleanClause.Occur.MUST);

		TopDocs t = searcher.search(nq, ir.numDocs());
		ScoreDoc[] hits = t.scoreDocs;
		Collections.shuffle(Arrays.asList(hits));
		System.out.println("Number of documents eligible for testing set: " + hits.length);

		// select the document ids from the global index to go into the testing index
		Set<Integer> testDocIDs = new HashSet<Integer>();
		for (int i = 0; i < numTesting; ++i) {
			int docID = (int) (Math.random() * hits.length);
			testDocIDs.add(hits[docID].doc);
		}

		int totalDocs = ir.numDocs();
		for (int docID = 0; docID < totalDocs; ++docID) {
			if (trainIW.numDocs() >= numTraining && testIW.numDocs() >= numTesting)
				break;
			Document doc = ir.document(docID);
			if (testDocIDs.contains(docID)) {
				testIW.addDocument(doc);
			}
			else {
				if (trainIW.numDocs() < numTraining)
					trainIW.addDocument(doc);
			}
		}
		ir.close();

		trainIW.commit();
		trainIW.close();

		testIW.commit();
		testIW.close();
	}
}
