#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <thread>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <limits>
#include <vector> 
#include <mutex>
#include "ttt.cpp"

using namespace std;

#define PORT 45000
#define SERVER_IP "127.0.0.1"

string global_nickname;

void showMenu() {
    cout << "\n===> Menú ===>\n"
         << "1) Ver Usuarios\n"
         << "2) Enviar Mensaje a Usuario\n"
         << "3) Mensaje a Todos\n"
         << "4) Enviar Archivo\n"
         << "5) Jugar TTT\n"
         << "6) Enviar Objeto Sala\n"
         << "7) Salir\n"
         << "Opción: " << flush;
}

string calculate_sha256_from_file(const string& filename) {
    string command = "sha256sum \"" + filename + "\" > temp_hash.txt";
    system(command.c_str());
    ifstream hash_file("temp_hash.txt");
    string hash_leido = "";
    if (hash_file.is_open()) {
        hash_file >> hash_leido;
        hash_file.close();
    }
    remove("temp_hash.txt");
    return hash_leido;
}

string formatLength(size_t len, int cifras) {
    stringstream ss;
    ss << setw(cifras) << setfill('0') << len;
    return ss.str();
}

int get_len(int clientSock, int n_prot){
    char readed[16];
    read(clientSock, readed, n_prot);
    readed[n_prot] = '\0';
    return atoi(readed);
}

string read_text(int clientSock, int len){
    char *readed = new char[len+1];
    read(clientSock, readed, len);
    readed[len] = '\0';
    string s(readed);
    delete[] readed;
    return s;
}

void registerNickname(int sockfd){
    int n;
    char buffer[2];
    struct timeval tv;
    tv.tv_sec = 1;   
    tv.tv_usec = 0;
    setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof tv);
    while (1) {
        cout << "Ingresa tu nombre de usuario: ";
        getline(cin, global_nickname);
        string nickMsg = string("n") + formatLength(global_nickname.size(), 2) + global_nickname;
        write(sockfd, nickMsg.c_str(), nickMsg.size());
        cout<<"Enviando al servidor ==> "<<nickMsg<<endl; 
        n = read(sockfd, buffer, 1);
        if (n <= 0) {
            cout << "\nNombre de usuario aceptado!\n\n";
            break;
        }
        buffer[n] = '\0';
        if (buffer[0] == 'E') {
            int len = get_len(sockfd, 3);
            string msg = read_text(sockfd, len);
            cout << "\nError msg => " << msg << "\n\n";
        }
    }
    struct timeval tv_reset;
    tv_reset.tv_sec = 0;
    tv_reset.tv_usec = 0;
    setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv_reset, sizeof tv_reset);   
}

void readThread(int socketConn) {
    char buffer[2];
    int n;
    while (true) {
        n = read(socketConn, buffer, 1);
        if (n <= 0) {
            cout << "\n[Server disconnected]\n";
            break;
        }
        buffer[n] = '\0';

        if (buffer[0] == 'L') {
            int len = get_len(socketConn, 2);
            cout<<endl<<"\n ---- Current Users ----\n";
            for(int i=0; i<len; i++){
                int len_nick = get_len(socketConn, 2);
                string nick = read_text(socketConn, len_nick);
                cout<<" - "<<nick;
                if(global_nickname==nick) cout<<" (tú)";
                cout<<endl;
            }
            showMenu();
        } 
        
        else if (buffer[0] == 'T') {
            // Mensaje privado (de usuario a usuario)
            int len = get_len(socketConn, 2);
            string from = read_text(socketConn, len);
            int len_msg = get_len(socketConn, 3);
            string msg = read_text(socketConn, len_msg);
            cout << "\n\nMENSAJE PRIVADO [" << from << ": " << msg << "]\n" << endl;
            showMenu();
        } 
        
        else if (buffer[0] == 'M') {
            // Mensaje broadcast (a todos los usuarios)
            int len = get_len(socketConn, 2);
            string from = read_text(socketConn, len);
            int len_msg = get_len(socketConn, 3);
            string msg = read_text(socketConn, len_msg);
            cout << "\n\nMENSAJE GLOBAL [" << from << ": " << msg << "]\n" << endl;
            showMenu();
        } 
        
        else if (buffer[0] == 'F') {
            int len_from = get_len(socketConn, 2);
            string from = read_text(socketConn, len_from);
            int len_fname = get_len(socketConn, 3);
            string fname = read_text(socketConn, len_fname);
            int fsize = get_len(socketConn, 10);
            string received_hash = read_text(socketConn, 64);

            cout << "\n[Archivo entrante de: " << from << "]" << endl;
            cout << " -> Nombre: " << fname << " (" << fsize << " bytes)" << endl;
            
            string base_name = fname;
            string extension = "";
            size_t dot_pos = fname.rfind('.');
            if (dot_pos != string::npos) {
                base_name = fname.substr(0, dot_pos);
                extension = fname.substr(dot_pos);
            }
            string save_as = base_name + "_" + global_nickname + extension;
            
            cout << " -> Guardando como: " << save_as << endl;
            
            FILE* fp = fopen(save_as.c_str(), "wb");
            char file_buffer[1024];
            int recvd = 0;
            while (recvd < fsize) {
                int r = read(socketConn, file_buffer, min(1024, fsize - recvd));
                if (r <= 0) break;
                fwrite(file_buffer, 1, r, fp);
                recvd += r;
            }
            fclose(fp);
            cout << "\n[Archivo recibido correctamente]" << endl;

            string calculated_hash = calculate_sha256_from_file(save_as);
            cout << " -> Hash recibido:    " << received_hash << endl;
            cout << " -> Hash calculado:   " << calculated_hash << endl;

            if (received_hash == calculated_hash) {
                cout << " -> [ÉXITO] El archivo está íntegro." << endl;
            } else {
                cout << " -> [ERROR] El archivo está corrupto. Los hashes no coinciden." << endl;
            }
            showMenu();
        } 
        
        else if(buffer[0]=='E') {
            int len = get_len(socketConn, 3);
            string msg = read_text(socketConn, len);
            cout <<"\n\nError del servidor => "<< msg << "\n\n";
            showMenu();
        } 

        else if (buffer[0] == 'R') {
            char roleChar;
            int r = read(socketConn, &roleChar, 1);
            if (r == 1) {
                {
                    lock_guard<mutex> lock(tttMutexClient);
                    tttRol = roleChar;
                    if (roleChar != 'O') {
                        tttHostEsperando = false;
                    }
                    if (roleChar == 'S') {
                        tttJuegoActivo = false;
                    }
                }
                if (roleChar == 'O') {
                    cout << "\n[TTT] Asignado como jugador O." << endl;
                } else if (roleChar == 'X') {
                    cout << "\n[TTT] Asignado como jugador " << roleChar << "." << endl;
                } else {
                    cout << "\n[TTT] Actualmente eres espectador." << endl;
                }
            }
        }

        else if (buffer[0] == 'H') {
            char flag;
            int r = read(socketConn, &flag, 1);
            if (r == 1) {
                {
                    lock_guard<mutex> lock(tttMutexClient);
                    tttHostEsperando = (flag == '1');
                }
                if (flag == '1') {
                    cout << "\n[TTT] Esperando a que otro usuario acepte la partida." << endl;
                } else {
                    cout << "\n[TTT] Oponente encontrado." << endl;
                }
            }
        }

        else if (buffer[0] == 'B') {
            string board = read_text(socketConn, 9);
            {
                lock_guard<mutex> lock(tttMutexClient);
                tttEstadoTablero = board;
            }
            cout << "\n[TTT] Actualización del tablero.";
            printTTTtablero(board);
        }

        else if (buffer[0] == 'U') {
            char turnChar;
            int r = read(socketConn, &turnChar, 1);
            if (r == 1) {
                char roleCopy;
                {
                    lock_guard<mutex> lock(tttMutexClient);
                    tttTurno = turnChar;
                    tttJuegoActivo = true;
                    roleCopy = tttRol;
                }
                if (roleCopy == turnChar) {
                    cout << "\n[TTT] Es tu turno (" << turnChar << "). Usa la opción 5 para jugar." << endl;
                } else {
                    cout << "\n[TTT] Turno de " << turnChar << "." << endl;
                }
            }
        }

        else if (buffer[0] == 'S') {
            int len = get_len(socketConn, 2);
            string challenger = read_text(socketConn, len);
            {
                lock_guard<mutex> lock(tttMutexClient);
                if (challenger != global_nickname) {
                    tttInvitacionDisponible = true;
                    tttInvitarNick = challenger;
                }
            }
            cout << "\n[TTT] " << challenger << " busca oponente. Usa la opción 5 para responder." << endl;
        }

        else if (buffer[0] == 'C') {
            char reason;
            int r = read(socketConn, &reason, 1);
            if (r == 1) {
                {
                    lock_guard<mutex> lock(tttMutexClient);
                    tttInvitacionDisponible = false;
                }
                if (reason == 'A') {
                    cout << "\n[TTT] Convocatoria atendida." << endl;
                } else {
                    cout << "\n[TTT] Convocatoria cancelada." << endl;
                    showMenu();
                }
            }
        }

        else if (buffer[0] == 'G') {
            int lenHost = get_len(socketConn, 2);
            string host = read_text(socketConn, lenHost);
            int lenOpp = get_len(socketConn, 2);
            string opp = read_text(socketConn, lenOpp);
            cout << "\n[TTT] Partida iniciada entre " << host << " y " << opp << "." << endl;
        }

        else if (buffer[0] == 'V') {
            char result;
            int r = read(socketConn, &result, 1);
            if (r == 1) {
                if (result == 'T') {
                    cout << "\n[TTT] Movimiento aceptado." << endl;
                } else {
                    cout << "\n[TTT] Movimiento inválido." << endl;
                }
            }
        }

        else if (buffer[0] == 'W') {
            char winner;
            int r = read(socketConn, &winner, 1);
            if (r == 1) {
                {
                    lock_guard<mutex> lock(tttMutexClient);
                    tttJuegoActivo = false;
                }
                cout << "\n[TTT] ¡Ganó " << winner << "!" << endl;
            }
        }

        else if (buffer[0] == 'D') {
            {
                lock_guard<mutex> lock(tttMutexClient);
                tttJuegoActivo = false;
            }
            cout << "\n[TTT] La partida terminó en empate." << endl;
        }

        else if (buffer[0] == 'Q') {
            char departed;
            int r = read(socketConn, &departed, 1);
            if (r == 1) {
                {
                    lock_guard<mutex> lock(tttMutexClient);
                    tttJuegoActivo = false;
                    tttHostEsperando = false;
                    tttInvitacionDisponible = false;
                }
                cout << "\n[TTT] La partida se canceló. El jugador " << departed << " se desconectó." << endl;
                showMenu();
            }
        }

        else if (buffer[0] == 'E') {
            char reason;
            int r = read(socketConn, &reason, 1);
            if (r == 1) {
                {
                    lock_guard<mutex> lock(tttMutexClient);
                    tttJuegoActivo = false;
                    tttHostEsperando = false;
                    tttInvitacionDisponible = false;
                    tttRol = 'S';
                    tttTurno = 'O';
                    tttEstadoTablero = "123456789";
                }
                if (reason == 'W') {
                    cout << "\n[TTT] La partida finalizó. Espera una nueva convocatoria." << endl;
                } else {
                    cout << "\n[TTT] La partida terminó en tablas. Espera una nueva convocatoria." << endl;
                }
            }
        }

        else if (buffer[0]=='O'){ 
            /*
            int from_len = get_len(socketConn, 2);
            string from_user = read_text(socketConn, from_len);
            
            cout << "\n\n<=== Objeto Recibido de [" << from_user << "] ===>" << endl;

            // --- Reconstruir el objeto leyendo los bytes ---
            sala sala_recibida;
            sala_recibida.c = new sala::cocina();

            read(socketConn, &sala_recibida.n, sizeof(sala_recibida.n));
            int len_str;
            read(socketConn, &len_str, sizeof(len_str));
            read(socketConn, sala_recibida.str, len_str);
            sala_recibida.str[len_str] = '\0';

            read(socketConn, &sala_recibida.c->cocinaNum, sizeof(sala_recibida.c->cocinaNum));
            int len_cocina_vec;
            read(socketConn, &len_cocina_vec, sizeof(len_cocina_vec));
            read(socketConn, sala_recibida.c->cocinaVector, len_cocina_vec);
            sala_recibida.c->cocinaVector[len_cocina_vec] = '\0';
            read(socketConn, &sala_recibida.c->cocinaFLotante, sizeof(sala_recibida.c->cocinaFLotante));

            // --- ¡Éxito! Imprimir el contenido del objeto ---
            cout << "  Contenido de la Sala:" << endl;
            cout << "    - n: " << sala_recibida.n << endl;
            cout << "    - str: '" << sala_recibida.str << "'" << endl;
            cout << "  Contenido de la Cocina:" << endl;
            cout << "    - cocinaNum: " << sala_recibida.c->cocinaNum << endl;
            cout << "    - cocinaVector: '" << sala_recibida.c->cocinaVector << "'" << endl;
            cout << "    - cocinaFlotante: " << static_cast<float>(sala_recibida.c->cocinaFLotante) << endl;
            cout << "<====================================>" << endl;
            
            delete sala_recibida.c;
            */
           showMenu();
        }  
    }
}

void sendFile(int sockfd) {
    string send_to, filepath;
    cout<<"\nEnviar archivo a (nickname): ";
    getline(cin, send_to);
    cout<<"Ruta del archivo: ";
    getline(cin, filepath);

    ifstream file_check(filepath);
    if (!file_check.good()) {
        cout << "Error: No se pudo abrir el archivo en la ruta: " << filepath << endl;
        return;
    }
    file_check.close();

    string file_hash = calculate_sha256_from_file(filepath);
    if (file_hash.empty() || file_hash.length() != 64) {
        cout << "Error: No se pudo calcular el hash. Asegúrate de que 'sha256sum' esté instalado en tu sistema." << endl;
        return;
    }
    
    ifstream file(filepath, ifstream::binary);
    file.seekg(0, file.end);
    long fsize = file.tellg();
    file.close();
    
    string filename = filepath;
    size_t last_slash = filepath.find_last_of("/\\");
    if (last_slash != string::npos) {
        filename = filepath.substr(last_slash + 1);
    }

    string header = string("f") 
                  + formatLength(send_to.size(), 2) + send_to
                  + formatLength(filename.size(), 3) + filename
                  + formatLength(fsize, 10)
                  + file_hash;

    cout<<" Enviando al servidor (cabecera) ==> "<<header<<endl; // <-- LÍNEA AÑADIDA
    write(sockfd, header.c_str(), header.size());

    FILE* fp = fopen(filepath.c_str(), "rb");
    char buffer[1024];
    while (!feof(fp)) {
        size_t n = fread(buffer, 1, sizeof(buffer), fp);
        if (n > 0) {
            write(sockfd, buffer, n);
        }
    }
    fclose(fp);
    cout<<"[Contenido del archivo enviado por completo: "<<filename<<"]\n";
}

int main(void) {
    struct sockaddr_in serverAddr;
    int sockfd;
    sockfd = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sockfd == -1) { perror("No se pudo crear el socket"); exit(EXIT_FAILURE); }

    memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(PORT);
    inet_pton(AF_INET, SERVER_IP, &serverAddr.sin_addr);
    connect(sockfd, (struct sockaddr*)&serverAddr, sizeof(serverAddr));

    cout << "Conectado al servidor en " << SERVER_IP << ":" << PORT << endl;
    registerNickname(sockfd);
    thread reader(readThread, sockfd);

    int opcion;
    string buf;
    showMenu();
    do {
        
        cin >> opcion;
        cin.ignore(numeric_limits<streamsize>::max(), '\n');
        
        if (opcion==1){
            char msg = 'l';
            write(sockfd, &msg, 1);
            cout<<"Enviando al servidor ==> "<<msg<<endl;
            this_thread::sleep_for(chrono::milliseconds(200));
            continue;
        }

        else if (opcion == 2){
            string send_to;
            cout<<"\nMensaje para: "; getline(cin, send_to);
            cout<<"\nEscribe el mensaje: "; getline(cin, buf);
            string msg = string("t") + formatLength(send_to.size(),2) + send_to + formatLength(buf.size(),3) + buf;
            write(sockfd, msg.c_str(), msg.size());
            cout<<"Enviando al servidor ==> "<<msg<<endl; // <-- LÍNEA AÑADIDA
            showMenu();
        }

        else if (opcion == 3){
            cout<<"\nMensaje para todos: "; getline(cin, buf);
            string msg = string("m") + formatLength(buf.size(),3) + buf;
            write(sockfd, msg.c_str(), msg.size());
            cout<<"Enviando al servidor ==> "<<msg<<endl; // <-- LÍNEA AÑADIDA
            showMenu();
        } 
        
        else if (opcion == 4){
            sendFile(sockfd);
            showMenu();
        } 

        else if (opcion == 5){
            char roleCopy;
            char turnCopy;
            string boardCopy;
            bool activeCopy;
            bool hostWaitingCopy;
            bool inviteCopy;
            string inviteNickCopy;

            {
                lock_guard<mutex> lock(tttMutexClient);
                roleCopy = tttRol;
                turnCopy = tttTurno;
                boardCopy = tttEstadoTablero;
                activeCopy = tttJuegoActivo;
                hostWaitingCopy = tttHostEsperando;
                inviteCopy = tttInvitacionDisponible;
                inviteNickCopy = tttInvitarNick;
            }

            if (roleCopy == 'O' && hostWaitingCopy) {
                cout << "\n[TTT] Sigues esperando a que otro usuario acepte la partida." << endl;
            }
            else if (!activeCopy && !hostWaitingCopy && inviteCopy && roleCopy != 'O') {
                string input;
                cout << "\n[TTT] " << inviteNickCopy << " busca oponente. ¿Aceptar? (s/n): ";
                getline(cin, input);
                if (!input.empty() && (input[0] == 's' || input[0] == 'S')) {
                    char acceptMsg = 'a';
                    write(sockfd, &acceptMsg, 1);
                    {
                        lock_guard<mutex> lock(tttMutexClient);
                        tttInvitacionDisponible = false;
                    }
                    cout << "[TTT] Has enviado la aceptación." << endl;
                } else {
                    cout << "[TTT] Invitación rechazada." << endl;
                }
            }
            else if (!activeCopy && !hostWaitingCopy && roleCopy != 'O') {
                char msg = 'p';
                write(sockfd, &msg, 1);
                cout << "\n[TTT] Solicitud enviada. Espera la convocatoria." << endl;
            }
            else if (activeCopy && (roleCopy == 'O' || roleCopy == 'X')) {
                cout << "\n[TTT] Tu rol actual: " << roleCopy << endl;
                printTTTtablero(boardCopy);

                if (turnCopy != roleCopy) {
                    cout << "[TTT] Turno actual: " << turnCopy << ". Espera tu turno." << endl;
                } else {
                    string input;
                    int pos = 0;
                    while (true) {
                        cout << "Ingresa la posición (1-9): ";
                        getline(cin, input);
                        if (input.size() == 1 && input[0] >= '1' && input[0] <= '9') {
                            pos = input[0] - '0';
                            break;
                        }
                        cout << "[TTT] Entrada inválida. Intenta de nuevo." << endl;
                    }

                    char moveMsg[2];
                    moveMsg[0] = 'g';
                    moveMsg[1] = static_cast<char>('0' + pos);
                    write(sockfd, moveMsg, 2);
                    cout << "[TTT] Movimiento enviado para la posición " << pos << "." << endl;
                }
            }
            else if (activeCopy) {
                cout << "\n[TTT] Partida en curso. Estás en modo espectador." << endl;
                printTTTtablero(boardCopy);
            }
            else {
                cout << "\n[TTT] No hay partida activa. Usa esta opción para iniciar una convocatoria." << endl;
            }

            showMenu();
        } 

        else if (opcion == 6){
            /*
            string send_to;
            cout << "\nEnviar objeto a (nickname): ";
            getline(cin, send_to);

            // --- Crear y poblar el objeto sala ---
            sala sala_obj;
            sala_obj.n = 101;
            strcpy(sala_obj.str, "Sala de Reuniones A-30");
            sala_obj.c = new sala::cocina();
            sala_obj.c->cocinaNum = 1;
            strcpy(sala_obj.c->cocinaVector, "Area de Cafe");
            sala_obj.c->cocinaFLotante = _Float16(99.9f);

            // --- Serializar y enviar ---
            char command = 'o'; // <--- CAMBIO a 'o' minúscula
            string to_msg = formatLength(send_to.size(), 2) + send_to;

            cout << "Enviando al servidor ==> " << command << to_msg << "[...datos binarios...]" << endl; // <-- LÍNEA AÑADIDA

            write(sockfd, &command, 1);
            write(sockfd, to_msg.c_str(), to_msg.size());

            int len_str = strlen(sala_obj.str);
            int len_cocina_vec = strlen(sala_obj.c->cocinaVector);
            write(sockfd, &sala_obj.n, sizeof(sala_obj.n));
            write(sockfd, &len_str, sizeof(len_str));
            write(sockfd, sala_obj.str, len_str);
            write(sockfd, &sala_obj.c->cocinaNum, sizeof(sala_obj.c->cocinaNum));
            write(sockfd, &len_cocina_vec, sizeof(len_cocina_vec));
            write(sockfd, sala_obj.c->cocinaVector, len_cocina_vec);
            write(sockfd, &sala_obj.c->cocinaFLotante, sizeof(sala_obj.c->cocinaFLotante));
            
            delete sala_obj.c;
            */
        }

        else if (opcion == 7) {
            char msg = 'x';
            write(sockfd, &msg, 1);
            cout<<" Enviando al servidor ==> "<<msg<<endl; // <-- LÍNEA AÑADIDA
            break;
        }
    } while (true);

    shutdown(sockfd, SHUT_RDWR);
    close(sockfd);
    reader.join();
    return 0;
}