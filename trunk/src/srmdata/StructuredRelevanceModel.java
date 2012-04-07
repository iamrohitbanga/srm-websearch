package srmdata;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

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

		public boolean equals(Object a, Object b) {
			TermDocPair td1 = (TermDocPair) a;
			TermDocPair td2 = (TermDocPair) b;
			return (td1.term.equals(td2.term) && td1.docID == td2.docID);
		}
	};

	class DocPair {
		int d1;
		int d2;
		public DocPair(int doc1, int doc2) {
			d1 = doc1;
			d2 = doc2;
		}
	};
	
	Map<TermDocPair, Double> probMat;
	Map<DocPair, Double> model;

	public StructuredRelevanceModel() {
		allDocIds = new HashSet<Integer>();
		testDocIds = new HashSet<Integer>();
		trainDocIds = new HashSet<Integer>();
		probMat = new HashMap<TermDocPair, Double>();
		model = new HashMap<DocPair, Double>();
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

	void computePriors(IndexReader ir, Set<Integer> testDocs, Set<Integer> modelDocIds, String fieldName) throws Exception {

		TermEnum terms = ir.terms();
		int collectionSize = 0;
		
		while (terms.next()) {

			Term t = terms.term();

			if (!t.field().equals(fieldName))
				continue;

			Set<Integer> allDocs = new TreeSet<Integer>(modelDocIds);

			int collectionFreq = 0;
			double pavg = 0;
			double meanfreq = 0;

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

			termDocs = ir.termDocs(t);
			while (termDocs.next()) {
				int d = termDocs.doc();

				if (!modelDocIds.contains(d))
					continue;

				int tf = termDocs.freq();
				int dl = ir.getTermFreqVector(d, fieldName).size();

				double R = (1.0 / (1.0 + meanfreq)) * Math.pow((meanfreq / (1.0+meanfreq)), tf);
				double pml = ((double)tf) / dl;
				double prob = Math.pow(pml, 1.0-R) * Math.pow(pavg, R);

				probMat.put(new TermDocPair(t.text(),d), prob);
				allDocs.remove(d);
			}
			termDocs.close();

			collectionFreq = -collectionFreq;
			// docs in which the term does not occur
			for (Integer d : allDocs) {
				probMat.put(new TermDocPair(t.text(),d), (double)collectionFreq);
			}
			++collectionSize;
//			if (collectionSize == 2257)
				System.out.println("Collection Size: "  + collectionSize + "   Probmat Size: " + probMat.size());
		}
		terms.close();

		System.out.println("here");
		
		terms = ir.terms();

		while (terms.next()) {

			Term t = terms.term();

			List<Integer> docsForTerm = new ArrayList<Integer>();
			TermDocs termDocs = ir.termDocs(t);
			while (termDocs.next())
				docsForTerm.add(termDocs.doc());

			for (Integer md : modelDocIds) {
				double val = probMat.get(new TermDocPair(t.text(), md));
				if (val < 0)
					val = -val/collectionSize;

				for (Integer q : testDocIds) {
					DocPair dp = new DocPair(q, md);
					Double prob = model.get(dp);
					if (prob == null) {
						prob = 1.0;
						model.put(dp, prob);
					}
					if (!docsForTerm.contains(q))
						val = 1.0 - val;
					assert (val >= 0.0 && val <= 1.0);
					prob = prob * val;
					model.put(dp, prob);
				}
			}

		}
		terms.close();
	}
}
