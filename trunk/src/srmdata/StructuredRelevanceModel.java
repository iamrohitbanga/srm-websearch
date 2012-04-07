package srmdata;

import java.io.File;
import java.lang.reflect.Array;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

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
	Set<Integer> testDocIds;
	Set<Integer> trainDocIds;

	class TermDocPair {
		String term;
		int docID;

		public TermDocPair(String t, int d) {
			term = t;
			docID = d;
		}

		@Override
		public boolean equals(Object a) {
			TermDocPair td1 = (TermDocPair) a;
			TermDocPair td2 = (TermDocPair) this;
			return (td1.term.equals(td2.term) && td1.docID == td2.docID);
		}
	};

	double[] modelScores;
	
	class DocPair {
		int d1;
		int d2;
		public DocPair(int doc1, int doc2) {
			d1 = doc1;
			d2 = doc2;
		}
		
		@Override
		public int hashCode() {
			int hash = 3571;
			hash = hash * 31 + d1;
			hash = hash * 31 + d2;
			return hash;
		}

		@Override
		public boolean equals(Object o) {
			return (d1 == ((DocPair) o).d1 && d2 == ((DocPair) o).d2);
		}
	};

	Map<DocPair, Double> model;

	public StructuredRelevanceModel() {
		allDocIds = new HashSet<Integer>();
		testDocIds = new HashSet<Integer>();
		trainDocIds = new HashSet<Integer>();
		model = new TreeMap<DocPair, Double>();
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

	void predictAudience() throws Exception {

		File nsdl_index_dir = new File(NSDLIndex.NSDL_INDEX_DIR_NAME);
		IndexReader ir = IndexReader.open(FSDirectory.open(nsdl_index_dir), true);

		computePriors(ir, testDocIds, trainDocIds, "title");

		ir.close();
	}

	boolean containsNumber(String str) {
		for (int i = 0; i < str.length(); ++i) {
			if (str.charAt(i) >= '0' && str.charAt(i) <= '9')
				return true;
		}
		return false;
	}
	
	
	void computePriors(IndexReader ir, Set<Integer> testDocs, Set<Integer> modelDocIds, String fieldName) throws Exception {

		Map<Integer, Integer> doc_length = new HashMap<Integer, Integer>();
		for (Integer docID : trainDocIds) {
			TermFreqVector tfVec = ir.getTermFreqVector(docID, "title");
			doc_length.put(docID, tfVec.size());
		}

		modelScores = new double[modelDocIds.size() * testDocIds.size()];
		for (int i = 0; i < modelScores.length; ++i) {
			modelScores[i] = 1.0;
		}
		
		int collectionSize = 0;
		TermEnum terms = ir.terms();
		while (terms.next()) {
			Term t = terms.term();
			if (!t.field().equals(fieldName) || containsNumber(t.text()))
				continue;
			collectionSize++;
		}
		terms.close();

		System.out.println("Collection Size: " + collectionSize);
		
		terms = ir.terms();
		while (terms.next()) {

			Term t = terms.term();
			if (!t.field().equals(fieldName) || containsNumber(t.text()))
				continue;

//			System.out.println("term: " + t.text());
			TermDocs termDocs = ir.termDocs(t);
			Set<Integer> docsForTerm = new HashSet<Integer>();
			while (termDocs.next())
				docsForTerm.add(termDocs.doc());
			termDocs.close();

			long t1 = System.nanoTime();
			Map<Integer, Double> mle = compute_mlestimate(ir, fieldName, modelDocIds, t, doc_length, collectionSize);
			DocPair dp = new DocPair(1, 1);
			long t2 = System.nanoTime();
			System.out.println("                Time Taken: " + ((double)(t2-t1)) / 1E9);
			for (Integer md : modelDocIds) {
				double val = mle.get(md);
				dp.d2 = md;
				for (Integer q : testDocs) {
					dp.d1 = q;
					Double prob = model.get(dp);
					if (prob == null) {
						prob = 1.0;
						model.put(dp, prob);
					}
					if (!docsForTerm.contains(q))
						val = 1.0 - val;
					prob = prob * val;
					model.put(dp, prob);
				}
			}
			long t3 = System.nanoTime();
			System.out.println("Time Taken: " + ((double)(t3-t2)) / 1E9);
		}
		terms.close();
	}

	private Map<Integer,Double> compute_mlestimate(IndexReader ir, String fieldName,
			Set<Integer> modelDocIds, Term t, Map<Integer, Integer> doc_length, int collectionSize) throws Exception {

		Double pavg = 0.0;
		Double meanfreq = 0.0;
		Integer collectionFreq = 0;
		compute_avgs(ir, fieldName, modelDocIds, t, pavg, meanfreq, collectionFreq);

		Map<Integer,Double> mlEstimates = new HashMap<Integer, Double>();
//		long t1 = System.nanoTime();

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
		
//		long t2 = System.nanoTime();
//		System.out.println("Time taken: " + ((double)(t2-t1)) / 1E9 );

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
