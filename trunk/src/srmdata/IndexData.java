package srmdata;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;

import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.Version;

public class IndexData {

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
				if (doc != null)
					iw.addDocument(doc);
				doc = new Document();
			}
			else {
				int index = line.indexOf(':');
				assert (index != -1);
				doc.add(new Field(line.substring(0,index), line.substring(index+2).getBytes()));
			}
		}
		System.out.println(iw.numDocs());
		iw.commit();
		iw.close();
		reader.close();
	}
}
