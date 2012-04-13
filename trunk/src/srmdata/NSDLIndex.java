package srmdata;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;

import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.Field.Index;
import org.apache.lucene.document.Field.Store;
import org.apache.lucene.document.NumericField;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.NumericRangeQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.NumericUtils;
import org.apache.lucene.util.Version;

public class NSDLIndex {

	public static String NSDL_FILE_NAME = "../../data/nsdl/nsdl.info";
	public static String NSDL_INDEX_DIR_NAME = "../../index/";
	public static String NSDL_GLOBAL_INDEX_DIR_NAME = "../../global_index/";
	public static Version VERSION = Version.LUCENE_35;

	public static void createSmallIndex() throws Exception {

		File nsdl_global_index_dir = new File(NSDL_GLOBAL_INDEX_DIR_NAME);
		IndexReader ir = IndexReader.open(FSDirectory.open(nsdl_global_index_dir), true);
		
		File nsdl_index_dir = new File(NSDL_INDEX_DIR_NAME);

		IndexWriterConfig iwConfig;
		StandardAnalyzer analyzer = new StandardAnalyzer(VERSION);
		iwConfig = new IndexWriterConfig(VERSION, analyzer);

		IndexWriter iw;
		iw = new IndexWriter(FSDirectory.open(nsdl_index_dir), iwConfig);
		iw.deleteAll();

		int totalDocs = ir.maxDoc();
		for (int i = 0; i < totalDocs; i++) {
			Document doc = ir.document(i);
			String audience = doc.get("audience");
			if (audience.equalsIgnoreCase("learner") || audience.equalsIgnoreCase("educator")) {
				if (Math.random() < 0.3)
					iw.addDocument(doc);
			}
			else {
				iw.addDocument(doc);
			}
		}
		
		ir.close();
		iw.commit();
		iw.close();
	}
	
	public static void createGlobalIndex() throws Exception {

		File nsdl_index_dir = new File(NSDL_GLOBAL_INDEX_DIR_NAME);

		IndexWriterConfig iwConfig;
		StandardAnalyzer analyzer = new StandardAnalyzer(VERSION);
		iwConfig = new IndexWriterConfig(VERSION, analyzer);

		IndexWriter iw;
		iw = new IndexWriter(FSDirectory.open(nsdl_index_dir), iwConfig);
		iw.deleteAll();

		BufferedReader reader = new BufferedReader(new FileReader(NSDL_FILE_NAME));

		Document doc = null;
		String line;

		int titleLen = 0;
		int contentLen = 0;
		int descLen = 0;
		int audienceLen = 0;
		int subjectLen = 0;
		int totalDocs = 0;
		
		while ((line = reader.readLine()) != null) {

			if (line.equals("")) {
				if (doc != null) {
					doc.add(new NumericField("title_len",
							NumericUtils.PRECISION_STEP_DEFAULT,
							Store.YES, true).setIntValue(titleLen));
					doc.add(new NumericField("content_len",
							NumericUtils.PRECISION_STEP_DEFAULT,
							Store.YES, true).setIntValue(contentLen));
					doc.add(new NumericField("desc_len",
							NumericUtils.PRECISION_STEP_DEFAULT,
							Store.YES, true).setIntValue(descLen));
					doc.add(new NumericField("audience_len",
							NumericUtils.PRECISION_STEP_DEFAULT,
							Store.YES, true).setDoubleValue(((double)audienceLen) / doc.getValues("audience").length));
					doc.add(new NumericField("subject_len",
							NumericUtils.PRECISION_STEP_DEFAULT,
							Store.YES, true).setDoubleValue(((double)subjectLen) / doc.getValues("subject").length));

					doc.add(new NumericField("num_audience",
							NumericUtils.PRECISION_STEP_DEFAULT,
							Store.YES, true).setIntValue(doc.getValues("audience").length));
					doc.add(new NumericField("num_educationLevel",
							NumericUtils.PRECISION_STEP_DEFAULT,
							Store.YES, true).setIntValue(doc.getValues("educationLevel").length));
					doc.add(new NumericField("num_subject",
							NumericUtils.PRECISION_STEP_DEFAULT,
							Store.YES, true).setIntValue(doc.getValues("subject").length));
					doc.add(new NumericField("num_sub",
							NumericUtils.PRECISION_STEP_DEFAULT,
							Store.YES, true).setIntValue(doc.getValues("sub").length));

					if (titleLen != 0 && contentLen != 0 && descLen != 0 &&
						doc.getValues("subject").length > 0 &&
						doc.getValues("audience").length == 1) {
					
						totalDocs++;
						iw.addDocument(doc);
					}

					titleLen = 0;
					contentLen = 0;
					descLen = 0;
					audienceLen = 0;
					subjectLen = 0;
				}
				doc = new Document();
			}
			else {
				int index = line.indexOf(':');
				assert (index != -1);
				String fieldName = line.substring(0,index);
				String fieldValue = line.substring(index+2);
				doc.add(new Field(fieldName, fieldValue, Store.YES, Index.ANALYZED, Field.TermVector.YES));
				if (fieldName.equals("title"))
					titleLen = fieldValue.length();
				else if (fieldName.equals("content"))
					contentLen = fieldValue.length();
				else if (fieldName.equals("desc"))
					descLen = fieldValue.length();
				else if (fieldName.equals("audience"))
					audienceLen += fieldValue.length();
				else if (fieldName.equals("subject"))
					subjectLen += fieldValue.length();
			}
		}

		System.out.println(iw.numDocs());
		analyzer.close();
		iw.commit();
		iw.close();
		reader.close();
	}

	public static void computeStatistics() throws Exception {
		File nsdl_index_dir = new File(NSDL_INDEX_DIR_NAME);
		IndexReader ir = IndexReader.open(FSDirectory.open(nsdl_index_dir), true);
		IndexSearcher searcher = new IndexSearcher(ir);

		NumericRangeQuery<Integer> nq1 = NumericRangeQuery.newIntRange("num_subject", 1, 100, true, true);
		NumericRangeQuery<Integer> nq2 = NumericRangeQuery.newIntRange("num_audience", 1, 1, true, true);
		NumericRangeQuery<Integer> nq3 = NumericRangeQuery.newIntRange("title_len", 1, 10000, true, true);
		NumericRangeQuery<Integer> nq4 = NumericRangeQuery.newIntRange("content_len", 20, 10000000, true, true);
		NumericRangeQuery<Integer> nq5 = NumericRangeQuery.newIntRange("desc_len", 1, 100000, true, true);

		BooleanQuery nq = new BooleanQuery();
		nq.add(nq1, BooleanClause.Occur.MUST);
		nq.add(nq2, BooleanClause.Occur.MUST);
		nq.add(nq3, BooleanClause.Occur.MUST);
		nq.add(nq4, BooleanClause.Occur.MUST);
		nq.add(nq5, BooleanClause.Occur.MUST);

		//		ScoreDoc[] hits = t.scoreDocs;
		TopDocs t = searcher.search(nq, 10);
		System.out.println("Total Hits:" + t.totalHits);
		searcher.close();
		ir.close();
	}
}
