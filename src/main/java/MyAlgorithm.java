import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class MyAlgorithm implements Classifier {
    // faza trenovavania instances
    //
    @Override
    public void buildClassifier(Instances data) throws Exception {

    }

    //vratit nieco ako index cielovej triedy
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return 0;
    }

    // mozno nic
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        return new double[0];
    }

    //kedy je ten algoritmus pouzitelny,
    // ked je to binarny klasifikator a su trie triedy exception
    // numeralne hodnoty vs lingvisticke
    @Override
    public Capabilities getCapabilities() {
        return null;
    }
}
