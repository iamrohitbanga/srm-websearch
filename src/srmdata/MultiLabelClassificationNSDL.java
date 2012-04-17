package srmdata;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
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
	private static final String OUTPUT_FILE_NAME = "../../outputs/output";
	private static int numTesting = 100;
	private static int numTraining = 20000;

	private static Set<Integer> testDocIDs;
	private static Set<Integer> trainDocIDs;

	private static Map<String,Double> time_taken;
	private static List<PredictionResult> results;
	private static final int MAX_RESULTS = 10;
	
	static class PredictionResult {
		int docID;
		Set<String> actualValues;
		LinkedHashSet<String> predictedValues;

		Set<String> actualTokens;
		List<List<String>> predictedTokens;
		
		PredictionResult() {
			actualValues = new HashSet<String>();
			predictedValues = new LinkedHashSet<String>();
			actualTokens = new LinkedHashSet<String>();
			predictedTokens = new ArrayList<List<String>>();
		}

		List<String> tokenize(String strValue) {
			List<String> tokens = new ArrayList<String>();
			String[] strs = strValue.split("[^a-zA-Z]");
			for (int i = 0; i < strs.length; ++i) {
				if (strs[i].length() > 2)
					tokens.add(strs[i]);
			}
			return tokens;
		}
		
		void addActualValue(String actualValue) {
			actualValue = actualValue.toLowerCase();
			actualValues.add(actualValue);
			actualTokens.addAll(tokenize(actualValue));
		}
		
		void addPredictedValue(String predictedValue) {
			predictedValue = predictedValue.toLowerCase();
			predictedValues.add(predictedValue);
			predictedTokens.add(tokenize(predictedValue));
		}

		public LinkedHashSet<String> getPredictedValues() {
			return predictedValues;
		}

		public Set<String> getActualValues() {
			return actualValues;
		}

		public List<List<String>> getPredictedTokenList() {
			return predictedTokens;
		}
		
		public boolean isRelevant(String token) {
			return actualTokens.contains(token);
		}
	}
	
	static {
		testDocIDs = new HashSet<Integer>();
		trainDocIDs = new HashSet<Integer>();
		time_taken = new LinkedHashMap<String,Double>();
		results = new ArrayList<PredictionResult>();
	}
	
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

		BufferedWriter writer = new BufferedWriter(new FileWriter(OUTPUT_FILE_NAME + numTesting + "_" + numTraining));
		dumpOutput(writer);
		writer.flush();
		writer.close();
	}

	private static void dumpOutput(BufferedWriter writer) throws IOException {

		writer.append("num_testing:" + numTesting + "\n");
		for (int docID : testDocIDs)
			writer.append(docID + ";");
		writer.append("\n");

		writer.append("num_training:" + numTraining + "\n");
		for (int docID : trainDocIDs)
			writer.append(docID + ";");
		writer.append("\n");

		writer.append("num_threads:" + PriorCalculator.numThreads + "\n");
		
		for (Map.Entry<String, Double> ttaken : time_taken.entrySet())
			writer.append("time_taken: " + ttaken.getKey() + " : " + ttaken.getValue() + "\n");

		double[] precisionAtK = new double[MAX_RESULTS];

		writer.append("prediction_results: " + numTesting + "\n");
		for (PredictionResult result : results) {

			writer.append("\n");
			writer.append("docID: " + result.docID + "\n");

			writer.append("actualValues: " + result.getActualValues().size() + "\n");
			for (String actualValue : result.getActualValues())
				writer.append(actualValue + "\n");

			writer.append("predictedValues: " + result.getPredictedValues().size() + "\n");
			for (String predictedValue : result.getPredictedValues())
				writer.append(predictedValue + "\n");

			List<List<String>> predictedTokenList = result.getPredictedTokenList();
			for (int rank = 1; rank <= predictedTokenList.size(); ++rank) {
				List<String> tokens = predictedTokenList.get(rank-1);
				int relevance = 0;
				for (String token : tokens) {
					if (result.isRelevant(token)) {
						relevance++;
					}
				}
				writer.append("\n");
				precisionAtK[rank-1] += ((double)relevance) / rank;
				for (int i = 0; i < precisionAtK.length; ++i) {
					writer.append(precisionAtK[i] + " ");
				}
				writer.append("\n");
			}
		}

		System.out.println("Precison@K: ");
		writer.append("Precison@K: ");
		for (int i = 0; i < precisionAtK.length; ++i) {
			precisionAtK[i] = precisionAtK[i] / results.size();
			System.out.print(precisionAtK[i] + " ");
			writer.append(precisionAtK[i] + " ");
		}
		System.out.println();
		writer.append("\n");
	}

	private static void predictfield(String fieldToPredict) throws Exception {

		RAMDirectory trainRAMDirectory = new RAMDirectory(FSDirectory.open(new File(TRAIN_INDEX_NAME))); 
		RAMDirectory testRAMDirectory = new RAMDirectory(FSDirectory.open(new File(TEST_INDEX_NAME))); 
		IndexReader trainIR = IndexReader.open(trainRAMDirectory, true);
		IndexReader testIR  = IndexReader.open(testRAMDirectory, true);

		int nTrainDocs = trainIR.numDocs();
		int nTestDocs = testIR.numDocs();

		long t1, t2;
		double ttaken;

		int num_fields = 3;
		double[][][] scores = new double[num_fields][][];
		StructuredRelevanceModel srm = new StructuredRelevanceModel();
		t1 = System.nanoTime();
//		scores[0] = srm.computePriors(testIR, trainIR, "title");
		PriorCalculator priorCalcTitle = new PriorCalculator(testIR, trainIR, "title");
		scores[0] = priorCalcTitle.computePriors();
		priorCalcTitle = null;
		t2 = System.nanoTime();
		ttaken = ((double)(t2-t1)) / 1E9;
		System.out.println("Time Taken Priors (title): " + ttaken);
		time_taken.put("title_model", ttaken);
		
		t1 = System.nanoTime();
//			scores[1] = srm.computePriors(testIR, trainIR, "desc");
		PriorCalculator priorCalcDesc = new PriorCalculator(testIR, trainIR, "desc");
		scores[1] = priorCalcDesc.computePriors();
		priorCalcDesc = null;
		t2 = System.nanoTime();
		ttaken = ((double)(t2-t1)) / 1E9;
		System.out.println("Time Taken Priors (desc): " + ttaken);
		time_taken.put("desc_model", ttaken);

		t1 = System.nanoTime();
//			scores[2] = srm.computePriors(testIR, trainIR, "content");
		PriorCalculator priorCalcContent = new PriorCalculator(testIR, trainIR, "content");
		scores[2] = priorCalcContent.computePriors();
		priorCalcContent = null;
		t2 = System.nanoTime();
		ttaken = ((double)(t2-t1)) / 1E9;
		System.out.println("Time Taken Priors (content): " + ttaken);
		time_taken.put("content_model", ttaken);

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
		for (int i = 0; i < nTestDocs; ++i) {
			double total_score = 0.0;
			for (int j = 0; j < nTrainDocs; ++j) {
				total_score += combined_score[i][j].score;
			}

			if (total_score == 0.0) {
				for (int j = 0; j < topN; ++j)
					combined_score[i][j].score = 0.0;
			}
			else {
				for (int j = 0; j < topN; ++j)
					combined_score[i][j].score /= total_score;
			}
		}
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

			PredictionResult result = new PredictionResult();
			System.out.print("docID:" + i + ":");
			for (Fieldable fieldable : testDoc.getFieldables(fieldToPredict)) {
				String actualValue = fieldable.stringValue();
				System.out.print(actualValue + ":");
				result.addActualValue(actualValue);
			}
			System.out.println(":");
			
			int max = (relevanceModel.size() < 10) ? relevanceModel.size() : 10;
			for (int j = 0; j < max; ++j) {
				Relevance relevance = relevanceModel.get(j);
				String predictedValue = relevance.fieldValue;
				System.out.println("    " + predictedValue + "(" + relevance.score + ");");
				result.addPredictedValue(predictedValue);
			}

			results.add(result);
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
				if (trainIW.numDocs() < numTraining) {
					trainDocIDs.add(docID);
					trainIW.addDocument(doc);
				}
			}
		}
		ir.close();

		trainIW.commit();
		trainIW.close();

		testIW.commit();
		testIW.close();
	}
}
