package srmdata;

public class SRMTester {

	
	public static void main(String[] args) throws Exception {
//		NSDLIndex.createIndex();
//		NSDLIndex.computeStatistics();
		
		StructuredRelevanceModel srm;
		srm = new StructuredRelevanceModel();
		srm.generateTestTrainSets_1();
	}
}
