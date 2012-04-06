package srmdata;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

class SRMRecord
{
	private List<String> educationLevel;
	private List<String> audience;
	private List<String> subject;
	private String description;
	private String content;
	private String title;
	
	SRMRecord()
	{
		audience = new ArrayList<String>();
		subject  = new ArrayList<String>();
		educationLevel = new ArrayList<String>();
	}
}

/*
 * Description - This class defines the data set for vocabulary
 */
class SRMVocabulary
{
	Map <String , Integer> contentVocabulary;
	Map <String , Integer> descriptionVocabulary;
	Map <String , Integer> audienceVocabulary;
	Map <String , Integer> subjectVocabulary;
	Map <String , Integer> titleVocabulary;
	Map <String , Integer> educationVocabulary;
}
public class SRM 
{
	public static void main(String [] args)
	{
		// Read and create training and testing records
		List <SRM> trainRecord = new ArrayList<SRM>();
		List <SRM> testRecord = new ArrayList<SRM>();
	}
}

