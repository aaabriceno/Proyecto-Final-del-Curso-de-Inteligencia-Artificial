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
#include <map>
#include <vector>
#include <algorithm> // [TTT] for std::count
#include <set>
#include <array>
#include <mutex>
//#include "ttt.cpp"

using namespace std;

#define PORT 45000
map<string, int> current_users;
map<int, string> current_users_ids;

void eliminarDeTTT(int clientSock) {
    vector<int> destinatarios;
    char departedRole = '?';
    bool notify = false;
    bool cancelSearch = false;

    {
        lock_guard<mutex> lock(tttMutexServer);

        if (tttGame.playerO == clientSock) {
            tttGame.playerO = -1;
            departedRole = 'O';
            notify = tttGame.active;
            if (tttGame.waitingForOpponent) {
                cancelSearch = true;
                tttGame.waitingForOpponent = false;
                tttGame.challengerSock = -1;
            }
            tttGame.active = false;
            tttGame.resetearTablero();
        } else if (tttGame.playerX == clientSock) {
            tttGame.playerX = -1;
            departedRole = 'X';
            notify = tttGame.active;
            tttGame.active = false;
            tttGame.resetearTablero();
        }

        tttGame.espectadores.erase(clientSock);

        if (notify) {
            destinatarios = reunirTTTDestinatariosBloqueados(tttGame);
            tttGame.resetearSesion();
        }
        if (tttGame.challengerSock == clientSock) {
            cancelSearch = true;
            tttGame.resetearSesion();
        }
    }

    if (notify && !destinatarios.empty()) {
        string msg = string("Q") + departedRole;
        tttBroadcast(destinatarios, msg);
    }
    if (cancelSearch) {
        string cancelMsg = string("C") + 'Q';
        for (const auto& entry : current_users) {
            write(entry.second, cancelMsg.c_str(), cancelMsg.size());
        }
    }
}

string longitudFormato(size_t len, int cifras) {
    stringstream ss;
    ss << setw(cifras) << setfill('0') << len;
    return ss.str();
}

int obtenerLongitud(int clientSock, int n_prot){
    char readed[16];
    read(clientSock, readed, n_prot);
    readed[n_prot] = '\0';
    return atoi(readed);
}

string leerTexto(int clientSock, int len){
    char *readed = new char[len+1];
    read(clientSock, readed, len);
    readed[len] = '\0';
    string s(readed);
    delete[] readed;
    return s;
}

void newClientThread(int clientSock) {
    char buffer[2];
    int n;

    while (true) {
        n = read(clientSock, buffer, 1);
        if (n <= 0) {
            eliminarDeTTT(clientSock);
            if(current_users_ids.count(clientSock)){
                cout << "\n[Cliente desconectado: " << current_users_ids[clientSock] << "]" << endl;
                current_users.erase(current_users_ids[clientSock]);
                current_users_ids.erase(clientSock);
            }
            break;
        }
        buffer[n] = '\0';
        string command_char(buffer); 

        if (buffer[0] == 'n') {
            int len = obtenerLongitud(clientSock, 2);
            string nick = leerTexto(clientSock, len);
            cout << " Recibido del cliente ==> " << command_char << longitudFormato(len,2) << nick << endl; // <-- LÍNEA AÑADIDA

            if (current_users.count(nick)){
                string msg = "Ese nickname ya existe.";
                string msg_error = string("E") + longitudFormato(msg.size(),3) + msg;
                cout << " Enviando al cliente ==> " << msg_error << endl; // <-- LÍNEA AÑADIDA
                write(clientSock, msg_error.c_str(), msg_error.size());
            } else {
                current_users[nick] = clientSock;
                current_users_ids[clientSock] = nick;
                cout << "[Nuevo usuario conectado: " << nick << "]" << endl;
            }
        } 

        else if (buffer[0] == 'm') {
            string from = current_users_ids[clientSock];
            int len_msg = obtenerLongitud(clientSock, 3); 
            string msg = leerTexto(clientSock, len_msg);
            cout << " Recibido del cliente ==> " << command_char << longitudFormato(len_msg,3) << msg << endl; // <-- LÍNEA AÑADIDA
            
            for(auto const &u : current_users){
                if (u.first != from){
                    string msg_list = string("M") + longitudFormato(from.size(),2) + from + longitudFormato(msg.size(),3) + msg;
                    cout << " Enviando al cliente (" << u.first << ") ==> " << msg_list << endl; // <-- LÍNEA AÑADIDA
                    write(u.second, msg_list.c_str(), msg_list.size());
                }
            }
        } 
        
        else if (buffer[0] == 't') {
            int len = obtenerLongitud(clientSock, 2);
            string to_send = leerTexto(clientSock, len);
            int len_msg = obtenerLongitud(clientSock, 3);
            string msg = leerTexto(clientSock, len_msg);
            string from = current_users_ids[clientSock];

            cout << " Recibido del cliente ==> " << command_char << longitudFormato(len,2) << to_send << longitudFormato(len_msg,3) << msg << endl; // <-- LÍNEA AÑADIDA

            if (!current_users.count(to_send)){
                string error_msg = "El nickname del destinatario no existe.";
                string msg_error = string("E") + longitudFormato(error_msg.size(),3) + error_msg;
                cout << " Enviando al cliente ==> " << msg_error << endl; // <-- LÍNEA AÑADIDA
                write(clientSock, msg_error.c_str(), msg_error.size());
            } else {
                string msg_to_send = string("T") + longitudFormato(from.size(), 2) + from + longitudFormato(msg.size(),3) + msg;
                cout << " Enviando al cliente (" << to_send << ") ==> " << msg_to_send << endl; // <-- LÍNEA AÑADIDA
                write(current_users[to_send], msg_to_send.c_str(), msg_to_send.size());
            }
        } 
        
        else if (buffer[0] == 'l') {
            cout << " Recibido del cliente ==> " << command_char << endl; // <-- LÍNEA AÑADIDA
            string msg = "";
            for(auto const &u : current_users){
                msg += longitudFormato(u.first.size(), 2) + u.first;
            }
            string msg_list = string("L") + longitudFormato(current_users.size(),2) + msg;
            cout << " Enviando al cliente ==> " << msg_list << endl; // <-- LÍNEA AÑADIDA
            write(clientSock, msg_list.c_str(), msg_list.size());

        } 
        
        else if (buffer[0] == 'f') { 
            int len_to = obtenerLongitud(clientSock, 2);
            string to_send = leerTexto(clientSock, len_to);
            int len_fname = obtenerLongitud(clientSock, 3);
            string fname = leerTexto(clientSock, len_fname);
            int fsize = obtenerLongitud(clientSock, 10);
            string file_hash = leerTexto(clientSock, 64);
            string from = current_users_ids[clientSock];

            cout << " Recibido del cliente (cabecera) ==> " << command_char << longitudFormato(len_to,2) << to_send << longitudFormato(len_fname,3) << fname << longitudFormato(fsize,10) << file_hash << endl; // <-- LÍNEA AÑADIDA
            cout << "Retransmitiendo archivo de " << from << " a " << to_send << endl;

            if (!current_users.count(to_send)) {
                string error_msg = "El destinatario no existe.";
                string msg_error = string("E") + longitudFormato(error_msg.size(),3) + error_msg;
                cout << " Enviando al cliente ==> " << msg_error << endl; // <-- LÍNEA AÑADIDA
                write(clientSock, msg_error.c_str(), msg_error.size());
                
                char temp_buf[1024];
                int recvd = 0;
                while (recvd < fsize) {
                    int r = read(clientSock, temp_buf, min(1024, fsize - recvd));
                    if (r <= 0) break;
                    recvd += r;
                }
            } 
            
            else {
                string header = string("F") + longitudFormato(from.size(),2) + from
                            + longitudFormato(fname.size(),3) + fname
                            + longitudFormato(fsize,10) + file_hash;
                cout << " Enviando al cliente (cabecera) ==> " << header << endl; // <-- LÍNEA AÑADIDA
                write(current_users[to_send], header.c_str(), header.size());

                char file_buffer[1024];
                int recvd = 0;
                while (recvd < fsize) {
                    int r = read(clientSock, file_buffer, min(1024, fsize - recvd));
                    if (r <= 0) break;
                    write(current_users[to_send], file_buffer, r);
                    recvd += r;
                }
                cout << " -> Retransmisión completada." << endl;
            }
        } 

        else if (buffer[0] == 'p') {
            cout << " Recibido del cliente ==> " << command_char << endl;

            string requesterNick = current_users_ids[clientSock];
            char assignedRole = 'S';
            bool sendBoardState = false;
            string boardSnapshot;
            char turnSnapshot = 'O';
            bool sendWaitingFlag = false;
            char waitingValue = '0';
            bool broadcastSearch = false;
            string searchNick;
            bool sendInviteInfo = false;
            string currentChallengerNick;

            {
                lock_guard<mutex> lock(tttMutexServer);

                if (tttGame.active) {
                    if (clientSock == tttGame.playerO) {
                        assignedRole = 'O';
                    } else if (clientSock == tttGame.playerX) {
                        assignedRole = 'X';
                    } else {
                        tttGame.espectadores.insert(clientSock);
                        assignedRole = 'S';
                    }
                    sendBoardState = true;
                    boardSnapshot = tttBoardString(tttGame.board);
                    turnSnapshot = tttGame.turn;
                } else if (tttGame.waitingForOpponent) {
                    if (clientSock == tttGame.challengerSock) {
                        assignedRole = 'O';
                        sendWaitingFlag = true;
                        waitingValue = '1';
                    } else {
                        tttGame.espectadores.insert(clientSock);
                        assignedRole = 'S';
                        sendInviteInfo = true;
                        currentChallengerNick = current_users_ids[tttGame.challengerSock];
                    }
                } else {
                    tttGame.resetearTablero();
                    tttGame.playerO = clientSock;
                    tttGame.playerX = -1;
                    tttGame.challengerSock = clientSock;
                    tttGame.waitingForOpponent = true;
                    tttGame.turn = 'O';
                    tttGame.espectadores.clear();
                    assignedRole = 'O';
                    sendWaitingFlag = true;
                    waitingValue = '1';
                    broadcastSearch = true;
                    searchNick = requesterNick;
                }
            }

            string roleMsg = string("R") + assignedRole;
            cout << " Enviando al cliente ==> " << roleMsg << endl;
            write(clientSock, roleMsg.c_str(), roleMsg.size());

            if (sendWaitingFlag) {
                string waitMsg = string("H") + waitingValue;
                write(clientSock, waitMsg.c_str(), waitMsg.size());
            }

            if (sendBoardState) {
                string boardMsg = string("B") + boardSnapshot;
                string turnMsg = string("U") + turnSnapshot;
                write(clientSock, boardMsg.c_str(), boardMsg.size());
                write(clientSock, turnMsg.c_str(), turnMsg.size());
            }

            if (sendInviteInfo) {
                int lenNick = static_cast<int>(currentChallengerNick.size());
                string inviteMsg = string("S") + longitudFormato(lenNick, 2) + currentChallengerNick;
                write(clientSock, inviteMsg.c_str(), inviteMsg.size());
            }

            if (broadcastSearch) {
                int lenNick = static_cast<int>(searchNick.size());
                string inviteMsg = string("S") + longitudFormato(lenNick, 2) + searchNick;
                for (const auto& entry : current_users) {
                    write(entry.second, inviteMsg.c_str(), inviteMsg.size());
                }
            }
        }

        else if (buffer[0] == 'a') {
            cout << " Recibido del cliente ==> " << command_char << endl;

            bool accepted = false;
            int playerOSock = -1;
            int playerXSock = -1;
            string boardSnapshot;
            char turnSnapshot = 'O';
            vector<int> destinatarios;
            string hostNick;
            string opponentNick;

            {
                lock_guard<mutex> lock(tttMutexServer);

                if (!tttGame.waitingForOpponent || tttGame.challengerSock == clientSock || tttGame.playerX != -1) {
                    accepted = false;
                } else {
                    tttGame.playerX = clientSock;
                    tttGame.waitingForOpponent = false;
                    tttGame.active = true;
                    tttGame.turn = 'O';
                    tttGame.resetearTablero();
                    tttGame.espectadores.clear();
                    for (const auto& entry : current_users) {
                        if (entry.second != tttGame.playerO && entry.second != tttGame.playerX) {
                            tttGame.espectadores.insert(entry.second);
                        }
                    }

                    boardSnapshot = tttBoardString(tttGame.board);
                    turnSnapshot = tttGame.turn;
                    destinatarios = reunirTTTDestinatariosBloqueados(tttGame);
                    playerOSock = tttGame.playerO;
                    playerXSock = tttGame.playerX;
                    hostNick = current_users_ids[playerOSock];
                    opponentNick = current_users_ids[playerXSock];
                    accepted = true;
                }
            }

            if (!accepted) {
                string denyMsg = string("V") + 'F';
                write(clientSock, denyMsg.c_str(), denyMsg.size());
                continue;
            }

            string clearInvite = string("C") + 'A';
            for (const auto& entry : current_users) {
                write(entry.second, clearInvite.c_str(), clearInvite.size());
            }

            if (playerOSock != -1) {
                string roleMsgO = string("R") + 'O';
                string waitOff = string("H") + '0';
                write(playerOSock, roleMsgO.c_str(), roleMsgO.size());
                write(playerOSock, waitOff.c_str(), waitOff.size());
            }

            if (playerXSock != -1) {
                string roleMsgX = string("R") + 'X';
                write(playerXSock, roleMsgX.c_str(), roleMsgX.size());
            }

            if (!destinatarios.empty()) {
                string boardMsg = string("B") + boardSnapshot;
                string turnMsg = string("U") + turnSnapshot;
                tttBroadcast(destinatarios, boardMsg);
                tttBroadcast(destinatarios, turnMsg);
            }

            int lenHost = static_cast<int>(hostNick.size());
            int lenOpp = static_cast<int>(opponentNick.size());
            string startMsg = string("G") + longitudFormato(lenHost, 2) + hostNick + longitudFormato(lenOpp, 2) + opponentNick;
            for (const auto& entry : current_users) {
                write(entry.second, startMsg.c_str(), startMsg.size());
            }
        }

        else if (buffer[0] == 'g') {
            char posBuf;
            int r = read(clientSock, &posBuf, 1);
            if (r != 1) {
                string validationMsg = string("V") + 'F';
                write(clientSock, validationMsg.c_str(), validationMsg.size());
                continue;
            }

            cout << " Recibido del cliente ==> " << command_char << posBuf << endl;

            int position = posBuf - '1';
            bool validMove = false;
            bool win = false;
            bool draw = false;
            bool announceNextTurn = false;
            char roleChar = '?';
            string boardSnapshot;
            char nextTurn = 'O';
            vector<int> destinatarios;
            bool endGame = false;
            char endReason = ' ';
            int playerOSock = -1;
            int playerXSock = -1;

            {
                lock_guard<mutex> lock(tttMutexServer);

                if (tttGame.playerO == clientSock) {
                    roleChar = 'O';
                } else if (tttGame.playerX == clientSock) {
                    roleChar = 'X';
                }

                if (roleChar == '?' || !tttGame.active || position < 0 || position >= 9 ||
                    tttGame.board[position] == 'O' || tttGame.board[position] == 'X' ||
                    tttGame.turn != roleChar) {
                    validMove = false;
                } else {
                    validMove = true;
                    tttGame.board[position] = roleChar;
                    boardSnapshot = tttBoardString(tttGame.board);
                    destinatarios = reunirTTTDestinatariosBloqueados(tttGame);

                    if (tttVerificarGanar(tttGame.board, roleChar)) {
                        win = true;
                        endGame = true;
                        endReason = 'W';
                        playerOSock = tttGame.playerO;
                        playerXSock = tttGame.playerX;
                        tttGame.resetearSesion();
                    } else if (tttTableroFull(tttGame.board)) {
                        draw = true;
                        endGame = true;
                        endReason = 'D';
                        playerOSock = tttGame.playerO;
                        playerXSock = tttGame.playerX;
                        tttGame.resetearSesion();
                    } else {
                        tttGame.turn = (tttGame.turn == 'O') ? 'X' : 'O';
                        nextTurn = tttGame.turn;
                        announceNextTurn = true;
                    }
                }
            }

            string validationMsg = string("V") + (validMove ? 'T' : 'F');
            cout << " Enviando al cliente ==> " << validationMsg << endl;
            write(clientSock, validationMsg.c_str(), validationMsg.size());

            if (validMove) {
                string boardMsg = string("B") + boardSnapshot;
                tttBroadcast(destinatarios, boardMsg);

                if (win) {
                    string winMsg = string("W") + roleChar;
                    tttBroadcast(destinatarios, winMsg);
                } else if (draw) {
                    string drawMsg = "D";
                    tttBroadcast(destinatarios, drawMsg);
                }

                if (announceNextTurn) {
                    string turnMsg = string("U") + nextTurn;
                    tttBroadcast(destinatarios, turnMsg);
                }
                if (endGame) {
                    if (playerOSock != -1) {
                        string roleReset = string("R") + 'S';
                        write(playerOSock, roleReset.c_str(), roleReset.size());
                    }
                    if (playerXSock != -1) {
                        string roleReset = string("R") + 'S';
                        write(playerXSock, roleReset.c_str(), roleReset.size());
                    }
                    string endMsg = string("E") + endReason;
                    tttBroadcast(destinatarios, endMsg);
                }
            }
        }

        else if (buffer[0] == 'x') {
            cout << " Recibido del cliente ==> " << command_char << endl; // <-- LÍNEA AÑADIDA
            cout << "\n[Usuario se ha desconectado: " << current_users_ids[clientSock] << "]" << endl;
            eliminarDeTTT(clientSock);
            current_users.erase(current_users_ids[clientSock]);
            current_users_ids.erase(clientSock);
            shutdown(clientSock, SHUT_RDWR);
            close(clientSock);
            break;
        } 
        
        else if (buffer[0] == 'o') { 
            /*
            string from = current_users_ids[clientSock];
            int len_to = obtenerLongitud(clientSock, 2);
            string to_send = leerTexto(clientSock, len_to);

            cout << "\nRecibido comando de objeto binario de [" << from << "] para [" << to_send << "]" << endl;
            
            // --- 1. Reconstruir el objeto leyendo los bytes ---
            sala sala_reconstruida;
            sala_reconstruida.c = new sala::cocina();

            // Leer miembros de 'sala'
            read(clientSock, &sala_reconstruida.n, sizeof(sala_reconstruida.n));
            int len_str;
            read(clientSock, &len_str, sizeof(len_str));
            read(clientSock, sala_reconstruida.str, len_str);
            sala_reconstruida.str[len_str] = '\0';

            // Leer miembros de 'cocina'
            read(clientSock, &sala_reconstruida.c->cocinaNum, sizeof(sala_reconstruida.c->cocinaNum));
            int len_cocina_vec;
            read(clientSock, &len_cocina_vec, sizeof(len_cocina_vec));
            read(clientSock, sala_reconstruida.c->cocinaVector, len_cocina_vec);
            sala_reconstruida.c->cocinaVector[len_cocina_vec] = '\0';
            read(clientSock, &sala_reconstruida.c->cocinaFLotante, sizeof(sala_reconstruida.c->cocinaFLotante));

            // --- 2. ¡Éxito! Imprimir en el servidor para verificar ---
            cout << "--- Objeto reconstruido en SERVIDOR ---" << endl;
            cout << "  sala.str: " << sala_reconstruida.str << endl;
            cout << "  cocina.cocinaVector: " << sala_reconstruida.c->cocinaVector << endl;
            cout << "------------------------------------" << endl;

            // --- 3. REENVIAR EL OBJETO AL CLIENTE DESTINO ---
            if (current_users.count(to_send)) { // Verifica si el usuario destino está conectado
                int dest_sock = current_users[to_send];
                
                // Protocolo: 'O' + [len_from(2)] + [from_user] + [datos binarios...]
                char command = 'O';
                string from_msg = longitudFormato(from.size(), 2) + from;

                cout << "Enviando objeto a [" << to_send << "] ==> " << command << from_msg << "[...datos binarios...]" << endl;

                // Escribir el comando y el remitente
                write(dest_sock, &command, 1);
                write(dest_sock, from_msg.c_str(), from_msg.size());

                // Escribir los datos binarios del objeto
                write(dest_sock, &sala_reconstruida.n, sizeof(sala_reconstruida.n));
                write(dest_sock, &len_str, sizeof(len_str));
                write(dest_sock, sala_reconstruida.str, len_str);
                write(dest_sock, &sala_reconstruida.c->cocinaNum, sizeof(sala_reconstruida.c->cocinaNum));
                write(dest_sock, &len_cocina_vec, sizeof(len_cocina_vec));
                write(dest_sock, sala_reconstruida.c->cocinaVector, len_cocina_vec);
                write(dest_sock, &sala_reconstruida.c->cocinaFLotante, sizeof(sala_reconstruida.c->cocinaFLotante));

            } else {
                cout << "Usuario [" << to_send << "] no encontrado. No se pudo reenviar el objeto." << endl;
            }

            // Limpieza de memoria
            delete sala_reconstruida.c;
            */
        }    
    }
}

int main(void) {
    struct sockaddr_in serverAddr;
    int serverSock, clientSock;
    vector<thread> threads;

    serverSock = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (serverSock == -1) { perror("No se pudo crear el socket"); exit(EXIT_FAILURE); }

    memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(PORT);
    serverAddr.sin_addr.s_addr = INADDR_ANY;

    if (bind(serverSock, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) == -1) {
        perror("Fallo en bind"); close(serverSock); exit(EXIT_FAILURE);
    }
    if (listen(serverSock, 10) == -1) {
        perror("Fallo en listen"); close(serverSock); exit(EXIT_FAILURE);
    }
    cout << "Servidor escuchando en el puerto " << PORT << "...\n";

    while (true) {
        clientSock = accept(serverSock, NULL, NULL);
        threads.emplace_back(newClientThread, clientSock);
    }
    for (auto& t : threads) {
        t.join();
    }
    close(serverSock);
    return 0;
}