import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.supervised.attribute.AttributeSelection;


import java.io.File;

@Getter
@Setter
@NoArgsConstructor

public class ArffManager {
    private String inputFileName;
    private String destinationFileName;
    private int classIndex;
    private int filteredClassIndex;


    public static void main(String[] args) throws Exception {
        String fileName = "src/main/resources/files/heart.csv";
        CSVLoader loader = new CSVLoader();
        loader.setFieldSeparator(",");
        loader.setSource(new File(fileName));
        Instances instances = loader.getDataSet();

        //select class column
        instances.setClassIndex(instances.numAttributes() - 1);

        // get the most relevant features in our model
        AttributeSelection attributeSelection = new AttributeSelection();

        //searching through the space of attributes subsets
        GreedyStepwise search = new GreedyStepwise();

        //get features with the most predictability
        CfsSubsetEval eval = new CfsSubsetEval();

        attributeSelection.setSearch(search);
        attributeSelection.setEvaluator(eval);
        attributeSelection.setInputFormat(instances);

        Instances filteredInstances = Filter.useFilter(instances, attributeSelection);

        System.out.println("Filtered dataset: ");
        System.out.println(filteredInstances.toSummaryString());

        //in classification model target variable should be a categorical variable or nominal variable
        NumericToNominal numericToNominal = new NumericToNominal();
        //index which starts with 1, you need to check parameter m_SelectedAttributes in attributeSelection
        numericToNominal.setAttributeIndices("10");
        numericToNominal.setInputFormat(filteredInstances);

        System.out.println("Processed dataset: ");
        Instances processedInstances = Filter.useFilter(filteredInstances, numericToNominal);
        System.out.println(processedInstances.toSummaryString());

        ArffSaver saver = new ArffSaver();
        saver.setInstances(processedInstances);
        saver.setFile(new File("src/main/resources/files/heart.arff"));
        saver.writeBatch();
    }


}
