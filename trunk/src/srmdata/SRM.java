package srmdata;

import java.io.File;
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

@SuppressWarnings("deprecation")
public class SRM 
{
	List<Integer> allDocIds;
	List<Integer> testDocIds;
	List<Integer> trainDocIds;
	
	public static int titleLength , contentLength, descLength;
	
	float titleWeight = 10f;
	float contentWeight = 1f;
	float descWeight = 1f;
		
	public Map <String , Integer> titleVocabMap;
	public Map <String , Integer> contentVocabMap;
	public Map <String , Integer> descVocabMap;
		
	public Map <Integer , Float> contentScore;
	public Map <Integer , Float> descriptionScore;
	public Map <Integer , Float> titleScore;
	
	public static Set<String> stopWords;
	public static Version VERSION = Version.LUCENE_CURRENT;
	
	public SRM() 
	{
		allDocIds = new ArrayList<Integer>();
		testDocIds = new ArrayList<Integer>();
		trainDocIds = new ArrayList<Integer>();
		
		titleVocabMap = new HashMap <String , Integer> ();
		contentVocabMap = new HashMap <String , Integer>();
		descVocabMap = new HashMap <String , Integer>();
		
		titleLength = 0 ;
		contentLength = 0; 
		descLength = 0;
	}
	
	public ImmutableSortedMap <Integer , Double> calculateDeltaKernelScore(Integer testDocId)
	{
		Set <String> rContent = new HashSet<String>();
		Set <String> rTitle = new HashSet<String>();
		Set <String> rDesc = new HashSet<String>();
	
		Map <String , Integer> titleMap;
		Map <String , Integer> descMap;
		Map <String , Integer> contentMap;
		
		String tempToken;
		
		int titleNI, contentNI, descNI;
		double titleSimScore , descSimScore, contentSimScore, simScore;
		
		Map <Integer , Double> scoreMap = new HashMap<Integer , Double>();
		ImmutableSortedMap <Integer , Double>sortedRankList = null;
		
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
		
		File nsdl_index_dir = new File(NSDLIndex.NSDL_GLOBAL_INDEX_DIR_NAME);
		IndexReader ir;
		try 
		{
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
				int docId = (Integer)trainIterator.next();
				doc = (Document)ir.document(docId);
								
				//********************** Title Similarity Score ***************
				ts = analyzer.tokenStream("title", new StringReader(doc.get("title")));
				termAtt = ts.addAttribute(TermAttribute.class);		
				
				// Construct a HashMap of Train record title
				titleMap = new HashMap<String , Integer>();
				titleNI = 0;
				
				while(ts.incrementToken())
				{
					tempToken = termAtt.term();
					if (tempToken.length() > 2)
					{
						titleNI++;
						if (titleMap.containsKey(tempToken))
						{
							titleMap.put(tempToken, titleMap.get(tempToken) + 1);
						}
						else
							titleMap.put(tempToken, 1);
					}
				}
				// Iterate over query title set to find similarity score
				iterator = rTitle.iterator();
				titleSimScore = 0.0;
				
				while(iterator.hasNext())
				{
					tempToken = iterator.next();
					if (titleMap.containsKey(tempToken))
					{
							titleSimScore += ((double)titleMap.get(tempToken) + (100 * titleVocabMap.get(tempToken)/titleLength))/(titleNI + 100);
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
					
					tempToken = termAtt.term();
					if (tempToken.length() > 2)
					{
						descNI++;
						if (descMap.containsKey(tempToken))
						{
							descMap.put(tempToken, descMap.get(tempToken) + 1);
						}
						else
							descMap.put(tempToken, 1);
					}
				}
								
				/// Iterate over query description set to find similarity score
				iterator = rDesc.iterator();
				descSimScore = 0.0;
				
				while(iterator.hasNext())
				{
					tempToken = iterator.next();
					if (descMap.containsKey(tempToken))
					{
						descSimScore += ((double)descMap.get(tempToken) + (100 * descVocabMap.get(tempToken))/descLength)/(descNI + 100);
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
					tempToken = termAtt.term();
					if (tempToken.length() > 2)
					{
						contentNI++;
						if (contentMap.containsKey(tempToken))
						{
							contentMap.put(tempToken, contentMap.get(tempToken) + 1);
						}
						else
							contentMap.put(tempToken, 1);
					}
				}
				
				// Iterate over query content set to find similarity score
				iterator = rContent.iterator();
				contentSimScore = 0.0;
				
				while(iterator.hasNext())
				{
					tempToken = iterator.next();
					if (contentMap.containsKey(tempToken))
					{
						contentSimScore += ((double)contentMap.get(tempToken) + 100 * contentVocabMap.get(tempToken) / contentLength)/(contentNI + 100);
					}
				}	
				
				//****************** Computer overall similarity score - Cross entropy **********************
				simScore = titleWeight * titleSimScore + contentWeight * contentSimScore + descWeight * descSimScore;
				scoreMap.put(docId , simScore);
								
				/* // Update the audience vocabulary Score
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
				}*/
			}
			
			// Sort documents by relevance
			Ordering valueComparator = Ordering.natural().onResultOf(Functions.forMap(scoreMap)).compound(Ordering.natural());
			sortedRankList = ImmutableSortedMap.copyOf(scoreMap, valueComparator);
		
		} 
		catch (Exception e) 
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return sortedRankList;
	}
	
	public void generateTestTrainSets() throws Exception 
	{

		File nsdl_index_dir = new File(NSDLIndex.NSDL_GLOBAL_INDEX_DIR_NAME);
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
		
		Document doc ;
		
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
		
		SnowballAnalyzer analyzer = new SnowballAnalyzer(SRM.VERSION , "Porter" , SRM.stopWords);
		TokenStream ts ;
		TermAttribute termAtt;
		String tempToken;
		
		Iterator trainIterator = trainDocIds.iterator();
		while (trainIterator.hasNext())
		{
			doc = (Document)ir.document((Integer)trainIterator.next());
			ts = analyzer.tokenStream("title", new StringReader(doc.get("title")));
			termAtt = ts.addAttribute(TermAttribute.class);		
			
			while(ts.incrementToken())
			{
				tempToken = termAtt.term();
				if (tempToken.length() > 2)
				{
					titleLength++;
					if (titleVocabMap.containsKey(tempToken))
					{
						titleVocabMap.put(tempToken, titleVocabMap.get(tempToken) + 1);
					}
					else
						titleVocabMap.put(tempToken, 1);
				}
			}
			
			ts = analyzer.tokenStream("content", new StringReader(doc.get("content")));
			termAtt = ts.addAttribute(TermAttribute.class);		
			
			while(ts.incrementToken())
			{
				tempToken = termAtt.term();
				if (tempToken.length() > 2)
				{
					contentLength++;
					tempToken = termAtt.term();
					if (contentVocabMap.containsKey(tempToken))
					{
						contentVocabMap.put(tempToken, contentVocabMap.get(tempToken) + 1);
					}
					else
						contentVocabMap.put(tempToken, 1);
				}
			}
			
			ts = analyzer.tokenStream("desc", new StringReader(doc.get("desc")));
			termAtt = ts.addAttribute(TermAttribute.class);		
			
			while(ts.incrementToken())
			{
				tempToken = termAtt.term();
				if (tempToken.length() > 2)
				{
					descLength++;
					tempToken = termAtt.term();
					if (descVocabMap.containsKey(tempToken))
					{
						descVocabMap.put(tempToken, descVocabMap.get(tempToken) + 1);
					}
					else
						descVocabMap.put(tempToken, 1);
				}
			}
		}
		
		/*System.out.println(titleLength + " " + contentLength + " " + descLength);
		System.out.println(" Title: " + titleVocabMap);
		System.out.println(contentVocabMap);
		System.out.println(descVocabMap);*/

		searcher.close();
		ir.close();
	}
	public static Double[] computePrecision(Object kArray[] , int K , int testDocId)
	{
		double precision = 0.0f , aPrecision , sPrecision , ePrecision;
		Set <String> rSubject = new HashSet<String>();
		Set <String> rAudience = new HashSet<String>();
		Set <String> rEduLevel = new HashSet<String>();
		aPrecision = sPrecision = ePrecision = 0;
		String tempToken;
		Double []result = new Double[3];
		
		Document doc;
		
		File nsdl_index_dir = new File(NSDLIndex.NSDL_GLOBAL_INDEX_DIR_NAME);
		IndexReader ir;
		try 
		{
			ir = IndexReader.open(FSDirectory.open(nsdl_index_dir), true);
			//IndexSearcher searcher = new IndexSearcher(ir);
			SnowballAnalyzer analyzer = new SnowballAnalyzer(SRM.VERSION , "Porter" , SRM.stopWords);
			TokenStream ts ;
			TermAttribute termAtt;
			
			// Query data
			doc = (Document)ir.document(testDocId);
			ts = analyzer.tokenStream("audience", new StringReader(doc.get("audience")));
			termAtt = ts.addAttribute(TermAttribute.class);	
			while (ts.incrementToken())
			{
				rAudience.add(termAtt.term());
			}
			
			ts = analyzer.tokenStream("subject", new StringReader(doc.get("subject")));
			termAtt = ts.addAttribute(TermAttribute.class);	
			while (ts.incrementToken())
			{
				rSubject.add(termAtt.term());
			}
			
			ts = analyzer.tokenStream("educationLevel", new StringReader(doc.get("educationLevel")));
			termAtt = ts.addAttribute(TermAttribute.class);	
			while (ts.incrementToken())
			{
				rEduLevel.add(termAtt.term());
			}
						
			for (int i = 0 ; i < K  ; i++)
			{
				int docId = (Integer)kArray[i];
				doc = (Document)ir.document(docId);
				
				ts = analyzer.tokenStream("audience", new StringReader(doc.get("audience")));
				termAtt = ts.addAttribute(TermAttribute.class);	
				
				
				while(ts.incrementToken())
				{
					tempToken = termAtt.term();
					if (tempToken.length() > 2 && rAudience.contains(tempToken))
					{
						aPrecision++;
						break;
					}
				}
				
				ts = analyzer.tokenStream("subject", new StringReader(doc.get("subject")));
				termAtt = ts.addAttribute(TermAttribute.class);	
				
				while(ts.incrementToken())
				{
					tempToken = termAtt.term();
					if (tempToken.length() > 2 && rSubject.contains(tempToken))
					{
						sPrecision++;
						break;
					}
				}
				
				ts = analyzer.tokenStream("educationLevel", new StringReader(doc.get("educationLevel")));
				termAtt = ts.addAttribute(TermAttribute.class);	
				while(ts.incrementToken())
				{
					tempToken = termAtt.term();
					if (tempToken.length() > 2 && rEduLevel.contains(tempToken))
					{
						ePrecision++;
						break;
					}
				}
				
			}
		}
		catch(Exception e)
		{
			System.out.println("Some exception occured");
		}
		//System.out.println("Audience : " + aPrecision/K + " Subject : " + sPrecision/K + " eduLevel : " + ePrecision/K);
		result[0] = aPrecision/K;
		result[1] = sPrecision/K;
		result[2] = ePrecision/K;
		return result;
	}
	public static void main(String [] args) 
	{
		SRM srmModel = new SRM();
		int count = 0;
		Iterator <Integer>iterator;
		List <Integer> rankedDocId;
		Object []array;
		int K = 20 , length;
		Object []kArray = new Object[K];
		double maPrecision = 0.0f , msPrecision = 0.0f , mePrecision = 0.0f ;
		Double result[];
		int testDocId;
				
		try 
		{
			NSDLIndex.constructStopWordSet();
			SRM.stopWords = NSDLIndex.stopWords;
			
			srmModel.generateTestTrainSets();
			
			// For each field - content, title, desc of all documents in testing set query the training set and find probability score of each documnet
			Iterator <Integer> testIterator = srmModel.testDocIds.iterator();
			while (testIterator.hasNext() && count < 20)
			{
				count++;
				rankedDocId = new ArrayList<Integer>();
				testDocId = (Integer)testIterator.next();
				iterator = srmModel.calculateDeltaKernelScore(testDocId).keySet().iterator();
				while (iterator.hasNext())
				{
					rankedDocId.add((Integer)iterator.next());
				}
				array = rankedDocId.toArray();
				length = rankedDocId.size();
				
				for(int counter = 0 ; counter < K ; counter++)
				{
					kArray[counter] = array[length - counter - 1];
				}
				result = computePrecision(kArray , K , testDocId);
				maPrecision += result[0];
				msPrecision += result[1];
				mePrecision += result[2];
			}
			System.out.println("MAP Audience : " + maPrecision/20);
			System.out.println("MAP Subject : " + msPrecision/20);
			System.out.println("MAP Education : " + mePrecision/20);
		} 
		catch (Exception e) 
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}

