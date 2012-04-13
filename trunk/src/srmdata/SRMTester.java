package srmdata;


public class SRMTester {

	
	public static void main(String[] args) throws Exception {
//		NSDLIndex.createGlobalIndex();
		NSDLIndex.createSmallIndex();
		NSDLIndex.computeStatistics();
//		StructuredRelevanceModel srm;
//		srm = new StructuredRelevanceModel();
//		srm.generateTestTrainSets_1();
//		srm.predictAudience();
	}
}
