package classifier.variants;

import weka.core.Instance;

public interface Variant {
    double[] distributionForInstance(Instance instance, int m_NumClasses);
    double classifyInstance(Instance instance, int m_NumClasses);
    String getOption();
}
