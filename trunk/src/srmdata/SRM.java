package srmdata;

import java.io.File;
import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.StringTokenizer;

import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.snowball.SnowballAnalyzer;
import org.apache.lucene.analysis.tokenattributes.TermAttribute;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.NumericRangeQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.Version;

import com.google.common.base.Functions;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Ordering;

/*
 * Analyze text and add to stop word list particularly special characters - remove numbers
 * To change - Ease constraints to include records with multiple audience, education level and subject in train/test data sets
 * Shuffle records into training and test sets randomly - include records with uniformly distributed audience, edu level and subject
 * improve tuning parameters - use term frequency in smoothing
 * Split by delimiters
 */

@SuppressWarnings("deprecation")
public class SRM 
{
	List<Integer> allDocIds;
	List<Integer> testDocIds;
	List<Integer> trainDocIds;
	
	float titleWeight = 100f;
	float contentWeight = 0.01f;
	float descWeight = 0.01f;
		
	Map <Integer , Float> contentScore;
	Map <Integer , Float> descriptionScore;
	Map <Integer , Float> titleScore;
	
	public static Set<String> stopWords;
	public static Version VERSION = Version.LUCENE_CURRENT;
	
	public SRM() 
	{
		allDocIds = new ArrayList<Integer>();
		testDocIds = new ArrayList<Integer>();
		trainDocIds = new ArrayList<Integer>();
	}
	
	public float calculateDeltaKernelScore(Integer testDocId)
	{
		float score = 0.0f;
		Set <String> rContent = new HashSet<String>();
		Set <String> rTitle = new HashSet<String>();
		Set <String> rDesc = new HashSet<String>();
	
		Map <String , Integer> titleMap;
		Map <String , Integer> descMap;
		Map <String , Integer> contentMap;
		
		String tempToken;
		
		int titleNI, contentNI, descNI;
		double titleSimScore , descSimScore, contentSimScore, simScore;
		
		Map <String , Double> audienceVocabularyScore = new HashMap<String , Double>();
		Map <String , Double> subjectVocabularyScore = new HashMap<String , Double>();
		Map <String , Double> educationVocabularyScore = new HashMap<String , Double>();
		
		String []predictedAudience;
		String []predictedSubject;
		StringTokenizer subjectVocabTokens , audienceVocabTokens;
		Set <String> subjectVocabSet , audienceVocabSet;
		
		Iterator <String> subjectVocabSetIterator , audienceVocabSetIterator;
		int i , length;
		
		Document doc;
		Iterator<String> iterator;
		
		File nsdl_index_dir = new File(NSDLIndex.NSDL_INDEX_DIR_NAME);
		IndexReader ir;
		try 
		{
			System.out.println("Test doc ID : " + testDocId);
			ir = IndexReader.open(FSDirectory.open(nsdl_index_dir), true);
			//IndexSearcher searcher = new IndexSearcher(ir);
			SnowballAnalyzer analyzer = new SnowballAnalyzer(SRM.VERSION , "Porter" , SRM.stopWords);
			TokenStream ts ;
			TermAttribute termAtt;
			
			// Query data
			doc = (Document)ir.document(testDocId);
			ts = analyzer.tokenStream("title", new StringReader(doc.get("title")));
			termAtt = ts.addAttribute(TermAttribute.class);	
			while (ts.incrementToken())
			{
				rTitle.add(termAtt.term());
			}
			
			ts = analyzer.tokenStream("content", new StringReader(doc.get("content")));
			termAtt = ts.addAttribute(TermAttribute.class);	
			while (ts.incrementToken())
			{
				rContent.add(termAtt.term());
			}
			
			ts = analyzer.tokenStream("desc", new StringReader(doc.get("desc")));
			termAtt = ts.addAttribute(TermAttribute.class);	
			while (ts.incrementToken())
			{
				rDesc.add(termAtt.term());
			}
						
			// Iterate over all training records to find the score of train , test document pair
			Iterator <Integer> trainIterator = trainDocIds.iterator();
			while (trainIterator.hasNext())
			{ 
				doc = (Document)ir.document((Integer)trainIterator.next());
								
				//********************** Title Similarity Score ***************
				ts = analyzer.tokenStream("title", new StringReader(doc.get("title")));
				termAtt = ts.addAttribute(TermAttribute.class);		
				
				// Construct a HashMap of Train record title
				titleMap = new HashMap<String , Integer>();
				titleNI = 0;
				
				while(ts.incrementToken())
				{
					titleNI++;
					tempToken = termAtt.term();
					
					if (titleMap.containsKey(tempToken))
					{
						titleMap.put(tempToken, titleMap.get(tempToken) + 1);
					}
					else
						titleMap.put(tempToken, 1);
				}
				
				// Iterate over query title set to find similarity score
				iterator = rTitle.iterator();
				titleSimScore = 0.0;
				
				while(iterator.hasNext())
				{
					tempToken = iterator.next();
					if (titleMap.containsKey(tempToken))
					{
						titleSimScore += ((double)titleMap.get(tempToken) + 1)/(titleNI + 2);
					}
					else
					{
						titleSimScore += 1.0 / (titleNI + 2);
					}
				}
				
				//********************** Description Similarity Score ****************
				ts = analyzer.tokenStream("desc", new StringReader(doc.get("desc")));
				termAtt = ts.addAttribute(TermAttribute.class);	
								
				// Construct a HashMap of Train record description
				descMap = new HashMap<String , Integer>();
				descNI = 0;
				
				while(ts.incrementToken())
				{
					descNI++;
					tempToken = termAtt.term();
					
					if (descMap.containsKey(tempToken))
					{
						descMap.put(tempToken, descMap.get(tempToken) + 1);
					}
					else
						descMap.put(tempToken, 1);
				}
								
				/// Iterate over query description set to find similarity score
				iterator = rDesc.iterator();
				descSimScore = 0.0;
				
				while(iterator.hasNext())
				{
					tempToken = iterator.next();
					if (descMap.containsKey(tempToken))
					{
						descSimScore += ((double)descMap.get(tempToken) + 1)/(descNI + 2);
					}
					else
					{
						descSimScore += 1.0 / (descNI + 2);
					}
				}
				
				//********************** Content Similarity Score ****************
				ts = analyzer.tokenStream("content", new StringReader(doc.get("content")));
				termAtt = ts.addAttribute(TermAttribute.class);	
								
				// Construct a HashMap of Train record content
				contentMap = new HashMap<String , Integer>();
				contentNI = 0;
				
				while(ts.incrementToken())
				{
					contentNI++;
					tempToken = termAtt.term();
					
					if (contentMap.containsKey(tempToken))
					{
						contentMap.put(tempToken, contentMap.get(tempToken) + 1);
					}
					else
						contentMap.put(tempToken, 1);
				}
				
				// Iterate over query content set to find similarity score
				iterator = rContent.iterator();
				contentSimScore = 0.0;
				
				while(iterator.hasNext())
				{
					tempToken = iterator.next();
					if (contentMap.containsKey(tempToken))
					{
						contentSimScore += ((double)contentMap.get(tempToken) + 1)/(contentNI + 2);
					}
					else
					{
						contentSimScore += 1.0 / (contentNI + 2);
					}
				}		
				//****************** Computer overall similarity score - Cross entropy **********************
				simScore = titleWeight * titleSimScore + contentWeight * contentSimScore + descWeight * descSimScore;
								
				// Update the audience vocabulary Score
				predictedAudience = doc.getValues("audience");
				length = predictedAudience.length;
				i = 0;
				
				// Construct vocabulary set for this doc
				audienceVocabSet = new HashSet<String>();
				while(i < length)
				{
					audienceVocabTokens = new StringTokenizer(predictedAudience[i], "':/");
					while(audienceVocabTokens.hasMoreTokens())
					{
						audienceVocabSet.add(audienceVocabTokens.nextToken().toLowerCase());
					}
					i++;
				}
				
				audienceVocabSetIterator = audienceVocabSet.iterator();
				while(audienceVocabSetIterator.hasNext())
				{
					tempToken = audienceVocabSetIterator.next();
					if (audienceVocabularyScore.containsKey(tempToken))
					{
						audienceVocabularyScore.put(tempToken, audienceVocabularyScore.get(tempToken) + simScore);
					}
					else
						audienceVocabularyScore.put(tempToken , simScore);
				}
				
				// Update the subject Vocabulary Score
				predictedSubject = doc.getValues("subject");
				length = predictedSubject.length;
				i = 0;
				
				// Construct vocabulary set for this doc
				subjectVocabSet = new HashSet<String>();
				while(i < length)
				{
					subjectVocabTokens = new StringTokenizer(predictedSubject[i], "':/");
					while(subjectVocabTokens.hasMoreTokens())
					{
						subjectVocabSet.add(subjectVocabTokens.nextToken().toLowerCase());
					}
					i++;
				}
				
				subjectVocabSetIterator = subjectVocabSet.iterator();
				while(subjectVocabSetIterator.hasNext())
				{
					tempToken = subjectVocabSetIterator.next();
					if (subjectVocabularyScore.containsKey(tempToken))
					{
						subjectVocabularyScore.put(tempToken, subjectVocabularyScore.get(tempToken) + simScore);
					}
					else
						subjectVocabularyScore.put(tempToken , simScore);
				}
			}
			// Display the sorted audience vocabulary
			/*Ordering valueComparator = Ordering.natural().onResultOf(Functions.forMap(audienceVocabularyScore)).compound(Ordering.natural());
			ImmutableSortedMap audienceVocabSortedMap = ImmutableSortedMap.copyOf(audienceVocabularyScore, valueComparator);
			System.out.println(audienceVocabSortedMap);*/ 
			
			// Display the sorted vocabulary
			Ordering valueComparator = Ordering.natural().onResultOf(Functions.forMap(subjectVocabularyScore)).compound(Ordering.natural());
			ImmutableSortedMap subjectVocabSortedMap = ImmutableSortedMap.copyOf(subjectVocabularyScore, valueComparator);
			System.out.println(subjectVocabSortedMap); 
		} 
		catch (Exception e) 
		{
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
		}
		Collections.shuffle(allDocIds);
		
		Iterator iterator = allDocIds.iterator();
		for (int i = 0; i < hits.length && iterator.hasNext(); ++i)
		{
			if (i < maxTrain)
				trainDocIds.add((Integer)iterator.next());
			else
				testDocIds.add((Integer)iterator.next());
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
			NSDLIndex.constructStopWordSet();
			SRM.stopWords = NSDLIndex.stopWords;
			
			srmModel.generateTestTrainSets();
			
			// For each field - content, title, desc of all documents in testing set query the training set and find probability score of each documnet
			Iterator <Integer> testIterator = srmModel.testDocIds.iterator();
			while (testIterator.hasNext() && count < 5)
			{
				count++;
				srmModel.calculateDeltaKernelScore((Integer)testIterator.next());
			}
		} 
		catch (Exception e) 
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}

