package srmdata;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.Field.Index;
import org.apache.lucene.document.Field.Store;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.Term;
import org.apache.lucene.queryParser.QueryParser;
import org.apache.lucene.search.Collector;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.Version;

public class NSDLIndex {

	public static String NSDL_FILE_NAME = "../../data/nsdl/nsdl.info";
	public static String NSDL_INDEX_DIR_NAME = "../../index/";
	public static Version VERSION = Version.LUCENE_35;

	public static void createIndex() throws Exception {

		File nsdl_index_dir = new File(NSDL_INDEX_DIR_NAME);

		IndexWriterConfig iwConfig;
		iwConfig = new IndexWriterConfig(VERSION, new StandardAnalyzer(VERSION));

		IndexWriter iw;
		iw = new IndexWriter(FSDirectory.open(nsdl_index_dir), iwConfig);
		iw.deleteAll();

		BufferedReader reader = new BufferedReader(new FileReader(NSDL_FILE_NAME));

		Document doc = null;
		String line;
		while ((line = reader.readLine()) != null) {

			if (line.equals("")) {
				if (doc != null) {
					doc.add(new Field("num_audience",
							"" + doc.getValues("audience").length,
							Store.YES, Index.NOT_ANALYZED));
					doc.add(new Field("num_educationLevel",
							"" + doc.getValues("educationLevel").length,
							Store.YES, Index.NOT_ANALYZED));
					doc.add(new Field("num_subject",
							"" + doc.getValues("subject").length,
							Store.YES, Index.NOT_ANALYZED));
					doc.add(new Field("num_sub",
							"" + doc.getValues("sub").length,
							Store.YES, Index.NOT_ANALYZED));
					iw.addDocument(doc);
				}
				doc = new Document();
			}
			else {
				int index = line.indexOf(':');
				assert (index != -1);
				doc.add(new Field(line.substring(0,index), line.substring(index+2), Store.YES, Index.ANALYZED));
			}
		}

		System.out.println(iw.numDocs());
		iw.commit();
		iw.close();
		reader.close();
	}

	public static void computeStatistics() throws Exception {
		File nsdl_index_dir = new File(NSDL_INDEX_DIR_NAME);
		IndexReader ir = IndexReader.open(FSDirectory.open(nsdl_index_dir), true);
		System.out.println("Commit User Data:" + ir.getCommitUserData());
		IndexSearcher searcher = new IndexSearcher(ir);
		QueryParser qp = new QueryParser(VERSION, "audience", new StandardAnalyzer(VERSION));
		TopDocs t = searcher.search(qp.parse("educationLevel:vocational"), 34063);
		ScoreDoc[] hits = t.scoreDocs;
		System.out.println("Total Hits:" + t.totalHits + " " + hits[0].doc + " " + hits[32000].doc);
	}
}
