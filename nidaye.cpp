#include <iostream>

using namespace std;

int main() {
	freopen("100data.txt","r",stdin);
	freopen("200data.txt","w",stdout);
	for (int i = 0; i < 145 * 145; i++) {
		double max=-1.0;
		int pos=0;
		double x;
		for (int j=0;j<16;j++) {
			cin>>x;
			if (x>max) {
				max=x;
				pos=j;
			}
		}
		cout<<pos<<endl;
	}
}