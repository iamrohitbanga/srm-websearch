package srmdata;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.NumericRangeQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.FSDirectory;

import com.google.common.base.Functions;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Ordering;

/*
 * To change - Ease constraints to include records with multiple audience, education level and subject in train/test data sets
 * Shuffle records into training and test sets randomly
 * Add education level for prediction
 * While creating indexes - choose the number of words to allow - stemming
 * Some basic NLP like stemming etc, all elements in lower case
 */

public class SRM 
{
	List<Integer> allDocIds;
	List<Integer> testDocIds;
	List<Integer> trainDocIds;
	
	float titleWeight = 0.9f;
	float contentWeight = 0.6f;
	float descWeight = 0.7f;
		
	Map <Integer , Float> contentScore;
	Map <Integer , Float> descriptionScore;
	Map <Integer , Float> titleScore;
	
	public SRM() 
	{
		allDocIds = new ArrayList<Integer>();
		testDocIds = new ArrayList<Integer>();
		trainDocIds = new ArrayList<Integer>();
	}
	
	public float calculateDeltaKernelScore(Integer testDocId)
	{
		float score = 0.0f;
		String rContent , rTitle, rDesc , wContent , wTitle , wDesc;
		Map <String , Integer> titleMap;
		Map <String , Integer> descMap;
		Map <String , Integer> contentMap;
		StringTokenizer sTokenizer;
		String tempToken;
		int titleNI, contentNI, descNI;
		double titleSimScore , descSimScore, contentSimScore, simScore;
		
		Map <String , Double> audienceVocabularyScore = new HashMap<String , Double>();
		Map <String , Double> subjectVocabularyScore = new HashMap<String , Double>();
		Map <String , Double> educationVocabularyScore = new HashMap<String , Double>();
				
		Document doc;
		
		File nsdl_index_dir = new File(NSDLIndex.NSDL_INDEX_DIR_NAME);
		IndexReader ir;
		try 
		{
			ir = IndexReader.open(FSDirectory.open(nsdl_index_dir), true);
			//IndexSearcher searcher = new IndexSearcher(ir);
			
			// Query data
			doc = (Document)ir.document(testDocId);
			rTitle = doc.get("title");
			rContent = doc.get("content");
			rDesc = doc.get("desc");
			
			System.out.println("Query audience : " + doc.get("audience"));
			System.out.println("Quer subject : " + doc.get("subject"));
			
			// Iterate over all training records to find the score of train , test document pair
			Iterator <Integer> trainIterator = trainDocIds.iterator();
			while (trainIterator.hasNext())
			{
				doc = (Document)ir.document((Integer)trainIterator.next());
				
				/********************** Title Similarity Score ****************/
				wTitle = doc.get("title");
				sTokenizer = new StringTokenizer(wTitle);
								
				// Construct a HashMap of Train record title
				titleMap = new HashMap<String , Integer>();
				titleNI = 0;
				
				while(sTokenizer.hasMoreTokens())
				{
					titleNI++;
					tempToken = sTokenizer.nextToken();
					
					if (titleMap.containsKey(tempToken))
					{
						titleMap.put(tempToken, titleMap.get(tempToken) + 1);
					}
					else
						titleMap.put(tempToken, 1);
				}
				
				//Tokenize the query title and find similarity score based on titles
				sTokenizer = new StringTokenizer(rTitle);
				titleSimScore = 0.0;
				
				while(sTokenizer.hasMoreTokens())
				{
					tempToken = sTokenizer.nextToken();
					if (titleMap.containsKey(tempToken))
					{
						titleSimScore += ((double)titleMap.get(tempToken) + 1)/(titleNI + 2);
					}
					else
					{
						titleSimScore += 1.0 / (titleNI + 2);
					}
				}
				
				/********************** Description Similarity Score ****************/
				wDesc = doc.get("desc");
				sTokenizer = new StringTokenizer(wDesc);
								
				// Construct a HashMap of Train record desc
				descMap = new HashMap<String , Integer>();
				descNI = 0;
				
				while(sTokenizer.hasMoreTokens())
				{
					descNI++;
					tempToken = sTokenizer.nextToken();
					
					if (descMap.containsKey(tempToken))
					{
						descMap.put(tempToken, descMap.get(tempToken) + 1);
					}
					else
						descMap.put(tempToken, 1);
				}
				
				//Tokenize the query description and find similarity score based on description
				sTokenizer = new StringTokenizer(rDesc);
				descSimScore = 0.f;
				
				while(sTokenizer.hasMoreTokens())
				{
					tempToken = sTokenizer.nextToken();
					if (descMap.containsKey(tempToken))
					{
						descSimScore += ((double)descMap.get(tempToken) + 1)/(descNI + 2);
					}
					else
					{
						descSimScore += 1.0 / (descNI + 2);
					}
				}
				
				/********************** Content Similarity Score ****************/
				wContent = doc.get("content");
				sTokenizer = new StringTokenizer(wContent);
								
				// Construct a HashMap of Train record content
				contentMap = new HashMap<String , Integer>();
				contentNI = 0;
				
				while(sTokenizer.hasMoreTokens())
				{
					contentNI++;
					tempToken = sTokenizer.nextToken();
					
					if (contentMap.containsKey(tempToken))
					{
						contentMap.put(tempToken, contentMap.get(tempToken) + 1);
					}
					else
						contentMap.put(tempToken, 1);
				}
				
				// Tokenize the query content and find similarity score based on content
				sTokenizer = new StringTokenizer(rContent);
				contentSimScore = 0;
				
				while(sTokenizer.hasMoreTokens())
				{
					tempToken = sTokenizer.nextToken();
					if (contentMap.containsKey(tempToken))
					{
						contentSimScore += ((double)contentMap.get(tempToken) + 1)/(contentNI + 2);
					}
					else
					{
						contentSimScore += 1.0 / (contentNI + 2);
					}
				}
				
				/****************** Computer overall similarity score - Cross entropy **********************/
				simScore = titleWeight * titleSimScore + contentWeight * contentSimScore + descWeight * descSimScore;
				//System.out.println("Sim " + simScore);
				
				// Update the audience vocabulary Score
				sTokenizer = new StringTokenizer(doc.get("audience"));
				while(sTokenizer.hasMoreTokens())
				{
					tempToken = sTokenizer.nextToken();
					if (audienceVocabularyScore.containsKey(tempToken))
					{
						audienceVocabularyScore.put(tempToken, audienceVocabularyScore.get(tempToken) + simScore);
					}
					else
						audienceVocabularyScore.put(tempToken , simScore);
				}
				
				// Update the subject Vocabulary Score
				sTokenizer = new StringTokenizer(doc.get("subject"));
				while(sTokenizer.hasMoreTokens())
				{
					tempToken = sTokenizer.nextToken();
					if (subjectVocabularyScore.containsKey(tempToken))
					{
						subjectVocabularyScore.put(tempToken, subjectVocabularyScore.get(tempToken) + simScore);
					}
					else
						subjectVocabularyScore.put(tempToken , simScore);
				}
				
			}
			// Display the sorted audience vocabulary
			Ordering valueComparator = Ordering.natural().onResultOf(Functions.forMap(audienceVocabularyScore)).compound(Ordering.natural());
			ImmutableSortedMap audienceVocabSortedMap = ImmutableSortedMap.copyOf(audienceVocabularyScore, valueComparator);
			System.out.println(audienceVocabSortedMap); 
			
			/* // Display the sorted vocabulary
			Ordering valueComparator = Ordering.natural().onResultOf(Functions.forMap(subjectVocabularyScore)).compound(Ordering.natural());
			ImmutableSortedMap subjectVocabSortedMap = ImmutableSortedMap.copyOf(subjectVocabularyScore, valueComparator);
			System.out.println(subjectVocabSortedMap); */
		} 
		catch (CorruptIndexException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} 
		catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
				
		return score;
	}
	
	public float calculatePCScore(String r , String w)
	{
		float score = 0.0f;
		return score;
	}
	
	public float calculateDirichletKernelScore(String r , String w)
	{
		float score = 0.0f;
		return score;
	}
	
	public void generateTestTrainSets() throws Exception {

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
		for (int i = 0; i < hits.length; ++i) 
		{
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
	
	public static void main(String [] args)
	{
		SRM srmModel = new SRM();
		int count = 0;
		try 
		{
			srmModel.generateTestTrainSets();
			
			// For each field - content, title, desc of all documents in testing set query the training set and find probability score of each documnet
			Iterator <Integer> testIterator = srmModel.testDocIds.iterator();
			
			while (testIterator.hasNext() && count < 100)
			{
				count++;
				srmModel.calculateDeltaKernelScore((Integer)testIterator.next());
				System.out.println(" Test Id:" + srmModel.testDocIds);
			}
		} 
		catch (Exception e) 
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}

