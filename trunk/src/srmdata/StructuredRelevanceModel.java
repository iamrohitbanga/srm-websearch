package srmdata;

import java.io.File;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.TermDocs;
import org.apache.lucene.index.TermEnum;
import org.apache.lucene.index.TermFreqVector;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.NumericRangeQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.FSDirectory;

public class StructuredRelevanceModel {

	Set<Integer> allDocIds;
	LinkedHashSet<Integer> testDocIds;
	Set<Integer> trainDocIds;

	
	public StructuredRelevanceModel() {
		allDocIds = new LinkedHashSet<Integer>();
		testDocIds = new LinkedHashSet<Integer>();
		trainDocIds = new LinkedHashSet<Integer>();
	}

	void generateTestTrainSets_1() throws Exception {

		File nsdl_index_dir = new File(NSDLIndex.NSDL_INDEX_DIR_NAME);
		IndexReader ir = IndexReader.open(FSDirectory.open(nsdl_index_dir), true);
		IndexSearcher searcher = new IndexSearcher(ir);

		NumericRangeQuery<Integer> nq1 = NumericRangeQuery.newIntRange("num_subject", 1, 100, true, true);
		NumericRangeQuery<Integer> nq2 = NumericRangeQuery.newIntRange("num_audience", 1, 1, true, true);
		NumericRangeQuery<Integer> nq3 = NumericRangeQuery.newIntRange("title_len", 1, 100000, true, true);
		NumericRangeQuery<Integer> nq4 = NumericRangeQuery.newIntRange("content_len", 1, 10000000, true, true);
		NumericRangeQuery<Integer> nq5 = NumericRangeQuery.newIntRange("desc_len", 1, 10000000, true, true);

		BooleanQuery nq = new BooleanQuery();
		nq.add(nq1, BooleanClause.Occur.MUST);
		nq.add(nq2, BooleanClause.Occur.MUST);
		nq.add(nq3, BooleanClause.Occur.MUST);
		nq.add(nq4, BooleanClause.Occur.MUST);
		nq.add(nq5, BooleanClause.Occur.MUST);

		TopDocs t = searcher.search(nq, 20000);
		ScoreDoc[] hits = t.scoreDocs;
		Collections.shuffle(Arrays.asList(hits));
		double testTrainRatio = 0.8;
		int maxTrain = (int) (testTrainRatio * hits.length);
		for (int i = 0; i < hits.length; ++i) {
			allDocIds.add(hits[i].doc);
			if (i < maxTrain)
				trainDocIds.add(hits[i].doc);
			else
				testDocIds.add(hits[i].doc);
		}

		System.out.println("Total Number of Documents: " + allDocIds.size());
		System.out.println("Total Number of Training Documents: " + trainDocIds.size());
		System.out.println("Total Number of Testing Documents: " + testDocIds.size());

		searcher.close();
		ir.close();
	}

	class Score {
		Double score;
		Integer docID;
	}

	void predictAudience() throws Exception {

		File nsdl_index_dir = new File(NSDLIndex.NSDL_INDEX_DIR_NAME);
		IndexReader ir = IndexReader.open(FSDirectory.open(nsdl_index_dir), true);

		int num_fields = 3;
		double[][] scores = new double[num_fields][]; 
		long t1 = System.nanoTime();
			 scores[0] = computePriors(ir, testDocIds, trainDocIds, "title");
		long t2 = System.nanoTime();
		System.out.println("                Time Taken: " + ((double)(t2-t1)) / 1E9);

			scores[1] = computePriors(ir, testDocIds, trainDocIds, "desc");
		long t3 = System.nanoTime();
		System.out.println("                Time Taken: " + ((double)(t3-t2)) / 1E9);

			scores[2] = computePriors(ir, testDocIds, trainDocIds, "content");
		long t4 = System.nanoTime();
		System.out.println("                Time Taken: " + ((double)(t4-t3)) / 1E9);

		Score[][] combined_score = new Score[testDocIds.size()][trainDocIds.size()];

		for (int i = 0; i < combined_score.length; ++i) {
			for (int j = 0; j < combined_score[i].length; ++j) {
				combined_score[i][j] = new Score();
			}
		}

		for (int i = 0; i < scores[0].length; ++i) {

			int x, y;
			x = i / testDocIds.size();
			y = i % testDocIds.size();

			combined_score[y][x].score = scores[0][i] * scores[1][i] * scores[2][i];
			combined_score[y][x].docID = x; 
//			System.out.println(" (" + i + "," + combined_score[i] + "," + scores[0][i]
//					+ "," + scores[1][i] + "," + scores[2][i] + ") ");
		}
//		System.out.println();

		for (int i = 0; i < testDocIds.size(); ++i) {
			Arrays.sort(combined_score[i], new DescendingComp());
		}

		Iterator<Integer> iter = testDocIds.iterator();
		for (int i = 0; i < 5; ++i) {
			System.out.println("for test doc: " + iter.next());
			for (int j = 0; j < 100; ++j) {
				System.out.println(" (" + i + "," + combined_score[i][j].docID + "," + combined_score[i][j].score + ") ");
			}
			System.out.println();
			System.out.println();
		}

		ir.close();
	}

	private static class DescendingComp implements Comparator<Score> {

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
	
	
	double[] computePriors(IndexReader ir, Set<Integer> testDocs, Set<Integer> modelDocs, String fieldName) throws Exception {

		Map<Integer, Integer> doc_length = new HashMap<Integer, Integer>();
		for (Integer docID : modelDocs) {
			TermFreqVector tfVec = ir.getTermFreqVector(docID, fieldName);
			doc_length.put(docID, tfVec.size());
		}

		Map<Integer, Integer> mdToSeqID = new HashMap<Integer, Integer>();
		Map<Integer, Integer> seqIDToMd = new HashMap<Integer, Integer>();
		int newMd = 0;
		for (int md : modelDocs) {
			mdToSeqID.put(md, newMd);
			seqIDToMd.put(newMd, md);
			newMd++;
		}

		Map<Integer, Integer> qToSeqID = new HashMap<Integer, Integer>();
		Map<Integer, Integer> seqIDToQ = new HashMap<Integer, Integer>();
		int newQ = 0;
		for (int q : testDocs) {
			qToSeqID.put(q, newQ);
			seqIDToQ.put(newQ, q);
			newMd++;
		}

		double[] modelScores;
		modelScores = new double[modelDocs.size() * testDocs.size()];
		for (int i = 0; i < modelScores.length; ++i)
			modelScores[i] = 1.0;

		int collectionSize = 0;
		TermEnum terms = ir.terms();
		while (terms.next()) {
			Term t = terms.term();
			if (!t.field().equals(fieldName) || containsNumber(t.text()))
				continue;
			TermDocs termDocs = ir.termDocs(t);
			while (termDocs.next()) {
				int d = termDocs.doc();
				if (!modelDocs.contains(d))
					continue;
				collectionSize += termDocs.freq();
			}
			termDocs.close();
		}
		terms.close();

		System.out.println("Collection Size: " + collectionSize);

		int[] docsForTerm = new int[testDocs.size()];
		
		terms = ir.terms();
		while (terms.next()) {

			Term t = terms.term();
			if (!t.field().equals(fieldName) || containsNumber(t.text()))
				continue;

			for (int i = 0; i < docsForTerm.length; ++i)
				docsForTerm[i] = 1;

			TermDocs termDocs = ir.termDocs(t);
			while (termDocs.next()) {
				if (!testDocs.contains(termDocs.doc()))
					continue;
				docsForTerm[qToSeqID.get(termDocs.doc())] = 0;
			}
			termDocs.close();

			Map<Integer, Double> mle = compute_mlestimate(ir, fieldName, modelDocs, t, doc_length, collectionSize);

			int mdSize = modelDocs.size();
			int qSize = testDocs.size();
			double[] val = new double[2];
			for (int md=0; md < mdSize; ++md) {
				val[0] = mle.get(seqIDToMd.get(md));
				val[1] = 1.0 - val[0];
				for (int q = 0; q < qSize; ++q) {
					modelScores[md*qSize + q] *= val[docsForTerm[q]];
				}
			}

		}
		terms.close();
		return modelScores;
	}

	private Map<Integer,Double> compute_mlestimate(IndexReader ir, String fieldName,
			Set<Integer> modelDocIds, Term t, Map<Integer, Integer> doc_length, int collectionSize) throws Exception {

		Double pavg = 0.0;
		Double meanfreq = 0.0;
		Integer collectionFreq = 0;
		compute_avgs(ir, fieldName, modelDocIds, t, pavg, meanfreq, collectionFreq);

		Map<Integer,Double> mlEstimates = new HashMap<Integer, Double>();

		TermDocs termDocs = ir.termDocs(t);
		while (termDocs.next()) {

			int d = termDocs.doc();
			if (!modelDocIds.contains(d))
				continue;

			int tf = termDocs.freq();
			double R = Math.pow((meanfreq/(1.0+meanfreq)),tf)/(1.0+meanfreq);
			double pml = ((double)tf) / doc_length.get(d);
			double val = Math.pow(pml, 1.0-R) * Math.pow(pavg, R);
			mlEstimates.put(d, val);
		}
		termDocs.close();

		double defaultVal = ((double)collectionFreq) / collectionSize;
		for (Integer md : modelDocIds) {
			if (!mlEstimates.containsKey(md))
				mlEstimates.put(md, defaultVal);
		}

		return mlEstimates;
	}

	private void compute_avgs(IndexReader ir, String fieldName,
			Set<Integer> modelDocIds, Term t,
			Double pavg, Double meanfreq, Integer collectionFreq) throws Exception {

		collectionFreq = 0;
		pavg = 0.0;
		meanfreq = 0.0;

		int count = 0;
		TermDocs termDocs = ir.termDocs(t);
		while (termDocs.next()) {

			int d = termDocs.doc();
			if (!modelDocIds.contains(d))
				continue;

			int tf = termDocs.freq();
			TermFreqVector tfv = ir.getTermFreqVector(d, fieldName);

			int dl = tfv.size();
			double pml = ((double)tf) / dl;
			pavg = pavg + pml;
			meanfreq = meanfreq + tf;

			collectionFreq = collectionFreq + tf;
			count++;
		}
		termDocs.close();

		pavg = pavg / count;
		meanfreq = meanfreq / count;
	}

}
